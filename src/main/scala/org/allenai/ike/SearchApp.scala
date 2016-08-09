package org.allenai.ike

import org.allenai.common.Config.EnhancedConfig
import org.allenai.common.Logging
import org.allenai.ike.persistence.Tablestore

import com.typesafe.config.Config
import nl.inl.blacklab.search.{ Hits, HitsWindow, Searcher, TextPattern }

import java.io.File
import java.util.concurrent.{ Callable, Executors, TimeUnit, TimeoutException }
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
case class SearchConfig(windowSize: Option[Int] = Some(100), evidenceLimit: Option[Int] = Some(1))
case class SearchRequest(query: Either[String, QExpr], target: Option[String],
  userEmail: Option[String], config: SearchConfig)
case class SearchResponse(qexpr: QExpr, groups: Seq[GroupedBlackLabResult])
case class CorpusDescription(name: String, description: Option[String])
case class SimilarPhrasesResponse(phrases: Seq[SimilarPhrase])

class SearchApp(val name: String, val description: Option[String], indexDir: File) extends Logging {
  logger.debug(s"Building SearchApp for $name")
  val searcher = Searcher.open(indexDir)

  def blackLabHits(textPattern: TextPattern, limit: Option[Int] = None) =
    SearchApp.blackLabHits(textPattern, searcher, limit)

  def fromHits(hitsIt: Iterator[Hits]) = SearchApp.fromHits(hitsIt, name)

  def semantics(query: QExpr): TextPattern = SearchApp.semantics(query)

  def search(qexpr: QExpr, searchConfig: SearchConfig) =
    SearchApp.search(qexpr, searcher, name, searchConfig)

  def search(qexpr: QExpr, windowSize: Option[Int]) =
    SearchApp.search(qexpr, searcher, name, windowSize)

  def wordAttributes(req: WordInfoRequest): Iterator[(String, String)] = {
    val textPattern = semantics(QWord(req.word))
    val hits = blackLabHits(textPattern, req.config.windowSize).take(1)
    val results = fromHits(hits)
    val data = results.flatMap(_.matchData)
    val attrs = data.flatMap(_.attributes.toSeq)

    attrs
  }
  def attrHist(attrs: Seq[(String, String)]): Map[(String, String), Int] =
    attrs.groupBy(identity).mapValues(_.size)
  def attrModes(attrs: Seq[(String, String)]): Map[String, String] = {
    val histogram = attrHist(attrs)
    val attrKeys = attrs.map(_._1).distinct
    val results = for {
      key <- attrKeys
      subHistogram = histogram.filterKeys(_._1 == key)
      if subHistogram.size > 0
      attrMode = subHistogram.keys.maxBy(subHistogram)
    } yield attrMode
    results.toMap
  }
  def wordInfo(req: WordInfoRequest): WordInfoResponse = {
    val attrs = wordAttributes(req).toSeq // TODO(michaels): maintain as iterable
    val histogram = attrHist(attrs)
    val modes = attrModes(attrs)
    val posTags = histogram.filterKeys(_._1 == "pos").map {
      case (a, b) => (a._2, b)
    }
    WordInfoResponse(req.word, posTags)
  }
}

object SearchApp extends Logging {
  def parse(r: SearchRequest): Try[QExpr] = parse(r.query, r.target.isDefined)

  def parse(q: Either[String, QExpr], allowCaptureGroups: Boolean = true): Try[QExpr] = q match {
    case Left(queryString) => QueryLanguage.parse(queryString, allowCaptureGroups)
    case Right(qexpr) => Success(qexpr)
  }

  /** Return hits from blacklab, paginized into windows.
    *
    * @param textPattern
    * @param windowSize
    * @return
    */
  def blackLabHits(textPattern: TextPattern, searcher: Searcher, windowSize: Option[Int] = None): Iterator[Hits] = {
    windowSize match {
      case Some(windowSize) =>
        // Paginate so we don't load everything into memory at once
        val results: Hits = searcher.find(textPattern)

        Iterator.from(0).map { i =>
          Try {
            val hits = results.window(i * windowSize, windowSize)
            logger.info(s"Returning batch #$i of size ${hits.size}")
            hits
          }
        }.takeWhile(_.isSuccess).map(_.get)
      case None =>
        // Don't paginate.  Return an iterator with a single Hit containing all results.
        Iterator(searcher.find(textPattern))
    }
  }

  def fromHits(hitsIt: Iterator[Hits], name: String): Iterator[BlackLabResult] =
    hitsIt.flatMap(hits => BlackLabResult.fromHits(hits, name))

  def semantics(query: QExpr): TextPattern = BlackLabSemantics.blackLabQuery(query)
  def search(
    qexpr: QExpr,
    searcher: Searcher,
    name: String,
    searchConfig: SearchConfig
  ): Iterator[BlackLabResult] = {
    val textPattern = semantics(qexpr)

    // We only want the grab results from the first window
    val hits = blackLabHits(textPattern, searcher: Searcher, searchConfig.windowSize)
    searchConfig.windowSize match {
      case Some(windowSize) => fromHits(hits, name).take(windowSize)
      case None => fromHits(hits, name)
    }
  }

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

  def apply(config: Config) = new SearchApp(config.getString("name"), config.get[String]("description"), DataFile.fromConfig(config))
}
