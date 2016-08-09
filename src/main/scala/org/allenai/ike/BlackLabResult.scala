package org.allenai.ike

import org.allenai.common.immutable.Interval

import nl.inl.blacklab.search.{ Hit, Hits, Kwic, Span }

import scala.collection.JavaConverters._
import scala.collection.mutable

// TODO: common.Interval isn't easily deserializable with Jackson, so this serves as a temporary
// replacement.
case class MyInterval(start: Int, end: Int) extends Ordered[MyInterval] {
  def shift(by: Int): MyInterval = MyInterval(this.start + by, this.end + by)

  override def compare(that: MyInterval): Int =
    if (this.start > that.start) {
      1
    } else if (this.start < that.start) {
      -1
    } else {
      this.length - that.length
    }

  def length: Int = end - start
}

case class BlackLabResult(
    wordData: Seq[WordData],
    matchOffset: MyInterval,
    captureGroups: Map[String, MyInterval],
    corpusName: String
) {
  def matchData: Seq[WordData] = wordData.slice(matchOffset.start, matchOffset.end)
  def matchWords: Seq[String] = matchData.map(_.word)
}

case object BlackLabResult {
  def wordData(hits: Hits, kwic: Kwic): Seq[WordData] = {
    val attrNames: mutable.Buffer[String] = kwic.getProperties.asScala
    val attrValues: Iterator[mutable.Buffer[String]] = kwic.getTokens.asScala.grouped(attrNames.size)
    val attrGroups = attrValues.map(attrNames.zip(_).toMap)
    (for {
      attrGroup <- attrGroups
      word = attrGroup.get("word") match {
        case Some(value) => value
        case _ => throw new IllegalStateException(s"kwic $kwic does not have 'word' attribute")
      }
      data = WordData(word, attrGroup - "word")
    } yield data).toSeq
  }

  def toInterval(span: Span): MyInterval = MyInterval(span.start, span.end)

  def captureGroups(hits: Hits, hit: Hit, shift: Int): Map[String, Option[MyInterval]] = {
    val names = hits.getCapturedGroupNames.asScala
    // For some reason BlackLab will sometimes return null values here, so wrap in Options
    val optSpans = hits.getCapturedGroups(hit) map wrapNull
    val result = for {
      (name, optSpan) <- names.zip(optSpans)
      optInterval = optSpan map toInterval
      shifted = optInterval map (_.shift(-shift))
    } yield (name, shifted)
    result.toMap
  }

  /** Converts a hit to a BlackLabResult. Returns None if the dreaded BlackLab NPE is returned
    * when computing the capture groups.
    */
  def fromHit(
    hits: Hits,
    hit: Hit,
    corpusName: String,
    kwicSize: Int = 20
  ): Option[BlackLabResult] = {
    val kwic = hits.getKwic(hit, kwicSize)
    val data = wordData(hits, kwic)
    val offset = MyInterval(kwic.getHitStart, kwic.getHitEnd)
    // TODO: https://github.com/allenai/okcorpus/issues/30
    if (hits.hasCapturedGroups) {
      val shift = hit.start - kwic.getHitStart
      val optGroups = captureGroups(hits, hit, shift)
      // If all of the capture groups are defined, then return the result. Otherwise, the
      // mysterious NPE was thrown and we can't compute the result's capture groups.
      if (optGroups.values.forall(_.isDefined)) {
        val groups = optGroups.mapValues(_.get)
        Some(BlackLabResult(data, offset, groups, corpusName))
      } else {
        None
      }
    } else {
      // No capture groups? No problem.
      Some(BlackLabResult(data, offset, Map.empty[String, MyInterval], corpusName))
    }
  }

  /** Converts the hits into BlackLabResult objects. If ignoreNpe is true, then it will skip over
    * any hits that throw the weird NPE. If ignoreNpe is false, will throw an IllegalStateException.
    */
  def fromHits(
    hits: Hits,
    corpusName: String,
    ignoreNpe: Boolean = true
  ): Iterator[BlackLabResult] = for {
    hit <- hits.iterator.asScala
    result <- fromHit(hits, hit, corpusName) match {
      case x if (x.isDefined || ignoreNpe) => x
      case _ => throw new IllegalStateException(s"Could not compute capture groups for $hit")
    }
  } yield result

  def wrapNull[A](a: A): Option[A] = if (a == null) None else Some(a)
}
