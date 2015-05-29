package org.allenai.dictionary.ml.subsample

import nl.inl.blacklab.search.lucene.SpanQueryCaptureGroup
import nl.inl.blacklab.search.sequences.SpanQuerySequence
import nl.inl.blacklab.search.{ Hits, Searcher }
import org.allenai.dictionary._
import org.allenai.dictionary.ml.{ GeneralizeToDisj, TokenizedQuery }
import org.apache.lucene.search.spans.SpanQuery

import scala.collection.JavaConverters._

object GeneralizedQuerySampler {

  def buildGeneralizedSpanQuery(
    qexpr: TokenizedQuery,
    searcher: Searcher,
    tables: Map[String, Table],
    sampleSize: Int
  ): SpanQuery = {
    def buildSpanQuery(qexpr: QExpr): SpanQuery = {
      searcher.createSpanQuery(
        BlackLabSemantics.blackLabQuery(QueryLanguage.interpolateTables(qexpr, tables).get)
      )
    }
    val generalizations = qexpr.generalizations.get
    
    // Build span queries for each query/generalization
    val generalizingSpanQueries = generalizations.zip(qexpr.getNamedTokens).map {
      case (GeneralizeToDisj(qpos, qsimiliar, _), (name, original)) =>
          val originalSq = buildSpanQuery(original)
          val extensions = (qpos ++ qsimiliar).map(buildSpanQuery)
          new SpanQueryTrackingDisjunction(originalSq, extensions, name)
      // Currently we do not handle GeneralizeToAll
      case (_, (name, original)) => buildSpanQuery(QNamed(original, name))
    }
    
    // Group the span queries into chunks and wrap the right chunks in capture groups
    var remaining = generalizingSpanQueries
    var chunked = List[SpanQuery]()
    qexpr.tokenSequences.foreach { ts =>
      val (chunk, rest) = remaining.splitAt(ts.size)
      remaining = rest
      val next = if (ts.isCaptureGroup) {
        new SpanQueryCaptureGroup(new SpanQuerySequence(chunk.asJava), ts.columnName.get)
      } else {
        new SpanQuerySequence(chunk.asJava)
      }
      chunked = next :: chunked
    }
    require(remaining.isEmpty)
    new SpanQuerySequence(chunked.reverse.asJava)
  }
}

/** Sampler that returns hits that could be matched by the generalizations of the input
  * query.
  *
  * @param maxEdits maximum edits a sentence can be from the query to be returned
  */
case class GeneralizedQuerySampler(maxEdits: Int, posSampleSize: Int)
    extends Sampler() {

  require(maxEdits >= 0)
  require(posSampleSize > 0)

  def buildGeneralizingQuery(tokenizedQuery: TokenizedQuery, searcher: Searcher,
    tables: Map[String, Table]): SpanQuery = {
    val gs = GeneralizedQuerySampler.buildGeneralizedSpanQuery(
      tokenizedQuery,
      searcher, tables, posSampleSize
    )
    if (tokenizedQuery.size < maxEdits) {
      new SpanQueryMinimumValidCaptures(gs, maxEdits, tokenizedQuery.getNames)
    } else {
      gs
    }
  }

  override def getSample(qexpr: TokenizedQuery, searcher: Searcher,
    targetTable: Table, tables: Map[String, Table]): Hits = {
    searcher.find(buildGeneralizingQuery(qexpr, searcher, tables))
  }

  override def getLabelledSample(
    qexpr: TokenizedQuery,
    searcher: Searcher,
    targetTable: Table,
    tables: Map[String, Table],
    startFromDoc: Int,
    startFromToken: Int
  ): Hits = {
    val rowQuery = Sampler.buildLabelledQuery(qexpr, targetTable)
    val rowSpanQuery = searcher.createSpanQuery(BlackLabSemantics.blackLabQuery(rowQuery))
    val sequenceQuery = buildGeneralizingQuery(qexpr, searcher, tables)
    searcher.find(new SpanQueryFilterByCaptureGroups(sequenceQuery, rowSpanQuery,
      targetTable.cols, startFromDoc, startFromToken))
  }
}
