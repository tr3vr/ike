package org.allenai.ike

import org.allenai.common.immutable.Interval

case class KeyedBlackLabResult(keys: Seq[Interval], result: BlackLabResult)

case class GroupedBlackLabResult(
  keys: Seq[String],
  size: Int,
  relevanceScore: Double,
  results: Iterable[KeyedBlackLabResult]
)
