package org.allenai.ike.ml

import org.allenai.ike._

import org.allenai.blacklab.search.Searcher

import scala.collection.JavaConverters._

object Generalization {
  def to(pos: Seq[QPos], phrase: Seq[QSimilarPhrases], partial: Boolean): Generalization = {
    if (pos.isEmpty && phrase.isEmpty) {
      GeneralizeToNone()
    } else {
      GeneralizeToDisj(pos, phrase, partial)
    }
  }
}

/** Represents a way of generalizing another QExpr*/
sealed abstract class Generalization()

/** Generalize to any token sequence of the given length */
// Note we currently do not handle this (we treat it as GeneralizeToNone)
// since my first attempt was very inefficient. The trouble is BlackLab can't optimize queries
// that involves capture groups of wildcards, might take a couple days to fix.
case class GeneralizeToAny(min: Int, max: Int) extends Generalization {
  require(min >= 0)
  require(max == -1 || max >= min)
}

/** Generalize to a query to match either itself of a different query from a fixed set.
  * Currently we only handle a very limited number of kinds of generalizations.
  * fullyGeneralizes is true if the generalizations are strictly more general then the
  * corresponding QExpr (so they could replace the QExpr)
  */
case class GeneralizeToDisj(pos: Seq[QPos], phrase: Seq[QSimilarPhrases], fullyGeneralizes: Boolean)
    extends Generalization {
  require(pos.nonEmpty || phrase.nonEmpty)
}

/** No generalizations possible */
case class GeneralizeToNone() extends Generalization

object QueryGeneralizer {

  // Map POS tags into groups of tags, so that if the user used a pos within a given group we
  // will consider suggesting all POS tags in that group as a suggestion
  val posSets = Seq(
    Set("VBZ", "VBP", "VBN", "VBG", "VBD", "VB"),
    Set("NNPS", "NN", "NNP", "NNS"),
    Set("PRP$", "PRP", "DT", "PDT", "EX", "MD", "LS"),
    Set("JJS", "JJR", "JJ", "RB", "IN", "DT", "PDT", "CC", "CD", "TO",
      "UH", "SYM", "POS", "PRP", "PDT", "EX", "MD", "LS"),
    Set("WRB", "WP$", "WDT", "WP"),
    Set("RBS", "RBR", "RP", "SYM", "RB", "IN", "CD", "MD")
  )

  private def getWordPosTags(
    qexpr: QExpr,
    searchers: Seq[Searcher],
    sampleSize: Int
  ): Seq[String] = {
    val posTags = searchers.flatMap { searcher =>
      val hits = searcher.find(BlackLabSemantics.blackLabQuery(qexpr)).window(0, sampleSize)
      hits.setContextSize(0)
      hits.setForwardIndexConcordanceParameters(null, null, List("pos").asJava)
      hits.asScala.map { hit =>
        val kwic = hits.getKwic(hit)
        val pos = kwic.getTokens("pos").get(0)
        pos
      }
    }
    posTags
  }

  /** Suggestion some generalizations for a given query expressions
    *
    * @param qexpr QExpr to generalize
    * @param searchers Searchers to use when deciding what to generalize
    * @param sampleSize Number of samples to get per a searcher when deciding what a word can be
    * generalized to
    * @return Generalization that could be made from the QExpr
    */
  def queryGeneralizations(
    qexpr: QExpr,
    searchers: Seq[Searcher],
    similarPhrasesSearcher: SimilarPhrasesSearcher,
    sampleSize: Int
  ): Generalization = {
    qexpr match {
      case QSimilarPhrases(words, pos, phrases) =>
        val pos = if (words.size == 1) {
          getWordPosTags(words.head, searchers, sampleSize).map(QPos)
        } else {
          Seq()
        }
        Generalization.to(pos, Seq(QSimilarPhrases(words, phrases.size, phrases)), true)
      case QWord(word) => // For words we sample the corpus for some possible POS tags
        val posTags = getWordPosTags(QWord(word), searchers, sampleSize)
        val similarPhrases = similarPhrasesSearcher.getSimilarPhrases(word)
        val qSimPhrases = if (similarPhrases.nonEmpty) {
          Seq(QSimilarPhrases(Seq(QWord(word)), similarPhrases.size, similarPhrases))
        } else {
          Seq()
        }
        val posTagCounts = posTags.groupBy(identity).mapValues(_.size)
        val minCountsThresh = Math.min(posTags.size / 20, 2)
        val posTagsToKeep = posTagCounts.filter(_._2 >= minCountsThresh).map(_._1)
        Generalization.to(posTagsToKeep.map(QPos).toSeq, qSimPhrases, true)
      case QPos(pos) =>
        val posTagsToUse = posSets.filter(_.contains(pos)).reduce(_ ++ _) - pos
        Generalization.to(posTagsToUse.map(QPos).toSeq, Seq(), false)
      case QDisj(qexprs) =>
        if (qexprs.size < 10) {
          val generalizations = qexprs.map(queryGeneralizations(_, searchers,
            similarPhrasesSearcher, sampleSize))
          if (generalizations.forall(!_.isInstanceOf[GeneralizeToAny])) {
            val candidates = generalizations.flatMap {
              case GeneralizeToDisj(pos, phrase, _) =>
                Some((pos, phrase))
              case _ => None
            }
            val (allPos, allPhrase) = if (candidates.nonEmpty) {
              candidates.reduce((g1, g2) => (g1._1 ++ g2._1, g1._2 ++ g2._2))
            } else {
              (Seq(), Seq())
            }
            val existingPos = qexprs.flatMap {
              case qp: QPos => Some(qp)
              case _ => None
            }
            Generalization.to(
              (allPos.toSet -- existingPos.toSet).toSeq,
              allPhrase, false
            )
          } else {
            val (min, max) = QueryLanguage.getQueryLength(qexpr)
            GeneralizeToAny(min, max)
          }
        } else {
          val (min, max) = QueryLanguage.getQueryLength(qexpr)
          GeneralizeToAny(min, max)
        }
      case _ => GeneralizeToNone()
    }
  }
}
