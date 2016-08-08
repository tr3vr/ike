package org.allenai.ike

import org.allenai.common.Config.EnhancedConfig
import org.allenai.common.Logging
import org.allenai.ike.ml.{ QuerySuggester, Suggestions }
import org.allenai.ike.persistence.Tablestore

import com.typesafe.config.Config
import org.allenai.blacklab.search.{ Hits, HitsWindow, Searcher, TextPattern }

import java.util.concurrent.{ Callable, Executors, TimeUnit }
import scala.util.control.NonFatal
import scala.util.{ Success, Try }

case class SuggestQueryRequest(
  query: String,
  userEmail: String,
  target: String,
  narrow: Boolean,
  config: SuggestQueryConfig
)
case class SuggestQueryConfig(beamSize: Int, depth: Int, maxSampleSize: Int, pWeight: Double,
  nWeight: Double, uWeight: Double)
case class ScoredStringQuery(query: String, score: Double, positiveScore: Double,
  negativeScore: Double, unlabelledScore: Double)
case class SuggestQueryResponse(
  original: ScoredStringQuery,
  suggestions: Seq[ScoredStringQuery],
  samplePercent: Double
)
case class WordInfoRequest(word: String, config: SearchConfig)
case class WordInfoResponse(word: String, posTags: Map[String, Int])
case class SearchConfig(limit: Int = 100, evidenceLimit: Int = 1)
case class SearchRequest(query: Either[String, QExpr], target: Option[String],
  userEmail: Option[String], config: SearchConfig)
case class SearchResponse(qexpr: QExpr, groups: Seq[GroupedBlackLabResult])
case class CorpusDescription(name: String, description: Option[String])
case class SimilarPhrasesResponse(phrases: Seq[SimilarPhrase])

case class SearchApp(config: Config) extends Logging {
  val name = config.getString("name")
  logger.debug(s"Building SearchApp for $name")
  val description = config.get[String]("description")
  val indexDir = DataFile.fromConfig(config)
  val searcher = Searcher.open(indexDir)

  def blackLabHits(textPattern: TextPattern, limit: Int): Try[HitsWindow] = Try {
    searcher.find(textPattern).window(0, limit)
  }

  def fromHits(hits: HitsWindow): Try[Seq[BlackLabResult]] = Try {
    BlackLabResult.fromHits(hits, name).toSeq
  }

  def semantics(query: QExpr): Try[TextPattern] = Try(BlackLabSemantics.blackLabQuery(query))

  def search(qexpr: QExpr, searchConfig: SearchConfig): Try[Seq[BlackLabResult]] = for {
    textPattern <- semantics(qexpr)
    hits <- blackLabHits(textPattern, searchConfig.limit)
    results <- fromHits(hits)
  } yield results

  def wordAttributes(req: WordInfoRequest): Try[Seq[(String, String)]] = for {
    textPattern <- semantics(QWord(req.word))
    hits <- blackLabHits(textPattern, req.config.limit)
    results <- fromHits(hits)
    data = results.flatMap(_.matchData)
    attrs = data.flatMap(_.attributes.toSeq)
  } yield attrs

  def attrHist(attrs: Seq[(String, String)]): Map[(String, String), Int] =
    attrs.groupBy(identity).mapValues(_.size)

  def attrModes(attrs: Seq[(String, String)]): Map[String, String] = {
    val histogram = attrHist(attrs)
    val attrKeys = attrs.map(_._1).distinct
    val results = for {
      key <- attrKeys
      subHistogram = histogram.filterKeys(_._1 == key)
      if subHistogram.nonEmpty
      attrMode = subHistogram.keys.maxBy(subHistogram)
    } yield attrMode
    results.toMap
  }

  def wordInfo(req: WordInfoRequest): Try[WordInfoResponse] = for {
    attrs <- wordAttributes(req)
    histogram = attrHist(attrs)
    modes = attrModes(attrs)
    posTags = histogram.filterKeys(_._1 == "pos").map {
      case (a, b) => (a._2, b)
    }
    res = WordInfoResponse(req.word, posTags)
  } yield res
}

object SearchApp extends Logging {
  def parse(r: SearchRequest): Try[QExpr] = r.query match {
    case Left(queryString) => QueryLanguage.parse(queryString, r.target.isDefined)
    case Right(qexpr) => Success(qexpr)
  }

  def parse(queryString: String, allowCaptureGroups: Boolean = true): Try[QExpr] = {
    QueryLanguage.parse(queryString, allowCaptureGroups)
  }

  def suggestQuery(
    searchApps: Seq[SearchApp],
    request: SuggestQueryRequest,
    similarPhrasesSearcher: SimilarPhrasesSearcher,
    timeoutInSeconds: Long
  ): Try[SuggestQueryResponse] = for {
    query <- QueryLanguage.parse(request.query)
    suggestion <- Try {
      val callableSuggestsion = new Callable[Suggestions] {
        override def call(): Suggestions = {
          QuerySuggester.suggestQuery(
            searchApps.map(_.searcher),
            query,
            Tablestore.tables(request.userEmail),
            Tablestore.namedPatterns(request.userEmail),
            similarPhrasesSearcher,
            request.target,
            request.narrow,
            request.config
          )
        }
      }
      val exService = Executors.newSingleThreadExecutor()
      val future = exService.submit(callableSuggestsion)
      try {
        future.get(timeoutInSeconds, TimeUnit.SECONDS)
      } catch {
        case NonFatal(e) =>
          logger.error(s"QuerySuggestor encountered exception: ${e.getMessage}")
          future.cancel(true) // Interrupt the suggestions
          throw e
      }
    }
    stringQueries <- Try(
      suggestion.suggestions.map(x => {
        ScoredStringQuery(QueryLanguage.getQueryString(x.query), x.score, x.positiveScore,
          x.negativeScore, x.unlabelledScore)
      })
    )
    original = suggestion.original
    originalString = ScoredStringQuery(QueryLanguage.getQueryString(original.query), original.score,
      original.positiveScore, original.negativeScore, original.unlabelledScore)
    totalDocs = searchApps.map(_.searcher.getIndexReader.numDocs()).sum
    response = SuggestQueryResponse(originalString, stringQueries,
      suggestion.docsSampledFrom / totalDocs.toDouble)
  } yield response

  /** Return hits from blacklab, paginated into windows.
    * @param textPattern the pattern to search for
    * @param searcher the searcher to execute queries against
    * @param windowSize if provided, the maximum number of hits to gather in one group
    * @return an iterator to the entire set of hits
    */
  def blackLabHits(textPattern: TextPattern, searcher: Searcher, windowSize: Option[Int] = None): Iterator[Hits] = {
    windowSize match {
      case Some(window) =>
        // Paginate so we don't load everything into memory at once
        val results: Hits = searcher.find(textPattern)

        Iterator.from(0).map { i =>
          Try {
            val hits = results.window(i * window, window)
            logger.info(s"Returning batch #$i of size ${hits.size}")
            hits
          }
        }.takeWhile(_.isSuccess).map(_.get)
      case None =>
        // Don't paginate.  Return an iterator with a single Hit containing all results.
        Iterator(searcher.find(textPattern))
    }
  }

  def fromHits(hitsIt: Iterator[Hits], name: String): Iterator[BlackLabResult] = {
    hitsIt.flatMap(hits => BlackLabResult.fromHits(hits, name))
  }

  def semantics(query: QExpr): TextPattern = BlackLabSemantics.blackLabQuery(query)

  def search(
    qexpr: QExpr,
    searcher: Searcher,
    name: String,
    windowSize: Option[Int]
  ): Iterator[BlackLabResult] = {
    val textPattern = semantics(qexpr)
    val hits = blackLabHits(textPattern, searcher, windowSize)
    fromHits(hits, name)
  }
}
