package org.allenai.ike.ml

import org.allenai.common.testkit.{ ScratchDirectory, UnitSpec }
import org.allenai.ike._
import org.allenai.ike.index.TestData
import org.allenai.ike.ml.Label._
import org.allenai.ike.ml.compoundop.{ EvaluatedOp, OpConjunction, OpConjunctionOfDisjunctions }
import org.allenai.ike.ml.queryop.{ QueryOp, SetToken }

import scala.collection.immutable.IntMap

class TestQuerySuggester extends UnitSpec with ScratchDirectory {
  TestData.createTestIndex(scratchDir)
  val searcher = TestData.testSearcher(scratchDir)
  val ss = new SimilarPhrasesSearcherStub()
  def buildMap(ints: Seq[Int]): IntMap[Int] = {
    IntMap[Int](ints.map((_, 0)): _*)
  }

  "SelectOperator" should "Select correct AND query" in {

    var matchId = 0
    def buildExample(k: Int): WeightedExample = {
      val label = if (k == 1) {
        Positive
      } else if (k == -1) {
        Negative
      } else {
        Unlabelled
      }
      matchId += 1
      WeightedExample(label, matchId, 0, 0, 1.0)
    }

    val examples = List(1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 0).
      map(buildExample).toIndexedSeq

    val op1 = SetToken(Prefix(1), QWord("p1"))
    val op2 = SetToken(Prefix(2), QWord("p2"))
    val op3 = SetToken(Prefix(3), QWord("p3"))

    val operatorHits = Map[QueryOp, IntMap[Int]](
      (op1, buildMap(List(1, 2, 3, 4, 5, 6, 7))),
      (op2, buildMap(List(1, 2, 3, 4, 5, 7, 8, 9))),
      (op3, buildMap(List(6, 7, 8, 9, 10)))
    )
    val data = HitAnalysis(operatorHits, examples)

    // OpConjunction and OpConjunctionOfDisjunction should be equivalent in this case
    Seq(
      (x: EvaluatedOp) => OpConjunction.apply(x),
      (x: EvaluatedOp) => OpConjunctionOfDisjunctions.apply(x)
    ).foreach { combiner =>
        val scoredQueries = QuerySuggester.selectOperator(
          data, new PositivePlusNegative(examples, -1),
          combiner,
          3, 2, 100
        )

        // Best Results is ANDing op1 and op2
        assertResult(Seq(
          Set(op1, op2),
          Set(op1),
          Set(op2)
        )) {
          scoredQueries.map(_._1.ops)
        }
      }
  }

  it should "Select correct OR queries" in {
    val examples = (Seq(Unlabelled) ++ Seq.fill(5)(Positive) ++
      Seq.fill(5)(Negative) ++ Seq(Unlabelled)).zipWithIndex.
      map(x => WeightedExample(x._1, x._2, 0, 0, 1.0)).toIndexedSeq

    val op1 = SetToken(Prefix(1), QWord("p1"))
    val op2 = SetToken(Prefix(2), QWord("p2"))
    val op3 = SetToken(Prefix(2), QWord("p3"))
    val op4 = SetToken(Prefix(3), QWord("p4"))

    val operatorHits = Map[QueryOp, IntMap[Int]](
      op1 -> buildMap(List(1, 2, 3)),
      op2 -> buildMap(List(3, 4, 7, 6, 8)),
      op3 -> buildMap(List(1, 2, 7, 6, 9)),
      op4 -> buildMap(List(1, 2, 3, 4, 5, 10, 11))
    )
    val data = HitAnalysis(operatorHits, examples)
    val scoredQueries = QuerySuggester.selectOperator(
      data, new PositivePlusNegative(examples, -2),
      (x: EvaluatedOp) => OpConjunctionOfDisjunctions(x),
      20, 4, 1000
    )

    // Best answer is (op2 OR op3) AND op4
    assertResult(Set(op2, op3, op4))(scoredQueries.head._1.ops)
    assertResult(4)(scoredQueries.head._2)
  }

  "QuerySuggester" should "Select diverse queries" in {
    val examples = (Seq(Unlabelled) ++ Seq.fill(5)(Positive) ++
      Seq.fill(5)(Negative) ++ Seq(Unlabelled)).zipWithIndex.
      map(x => WeightedExample(x._1, x._2, 0, 0, 1.0)).toIndexedSeq

    val op1 = SetToken(Prefix(1), QWord("p1"))
    val op2 = SetToken(Prefix(2), QWord("p2"))
    val op3 = SetToken(Prefix(3), QWord("p3"))

    val operatorHits = Map[QueryOp, IntMap[Int]](
      op1 -> buildMap(List(1, 2, 3)),
      op2 -> buildMap(List(1, 2, 6)),
      op3 -> buildMap(List(1, 3, 6, 7))
    )
    val data = HitAnalysis(operatorHits, examples)

    Seq(2, 3).foreach {
      case i =>
        val scoredQueries = QuerySuggester.selectOperator(
          data, new PositivePlusNegative(examples, -2),
          (x: EvaluatedOp) => OpConjunctionOfDisjunctions(x),
          20, 3, i
        )
        // op1 OR op2 is better then op2, but we should not suggest if i == 2 since
        // then op1 would be over used
        val expected = Seq(Set(op1), Set(op1, op3), Set(op2))
        val actual = scoredQueries.take(3).map(_._1.ops)
        if (i == 2) assert(expected == actual) else assert(expected != actual)
    }
  }

  it should "suggest POS generalization" in {
    val table = Table("test", Seq("c1"),
      Seq(Seq("i"), Seq("like"), Seq("hate")).map { x =>
        TableRow(x.map(y => TableValue(Seq(QWord(y)))))
      },
      Seq())
    val startingQuery = QueryLanguage.parse("I (?<c1> like)").get
    val suggestions =
      QuerySuggester.suggestQuery(Seq(searcher), startingQuery, Map("test" -> table), Map(), ss,
        "test", false, SuggestQueryConfig(5, 2, 100, 10, 0, -0.1))
    assertResult((1, 0, 0))((suggestions.original.positiveScore, suggestions.original.negativeScore,
      suggestions.original.unlabelledScore))
    val bestSuggestions = suggestions.suggestions.take(2).map(_.query)
    val best = suggestions.suggestions.head
    assertResult((2, 0, 0))((best.positiveScore, best.negativeScore, best.unlabelledScore))
    assert(bestSuggestions contains QueryLanguage.parse("PRP (?<c1> VBP)").get)
    assert(bestSuggestions contains QueryLanguage.parse("I (?<c1> VBP)").get)
  }

  it should "suggest removing plus operators" in {
    val table = Table("test", Seq("c1"),
      Seq(Seq("i"), Seq("it")).map { x =>
        TableRow(x.map(y => TableValue(Seq(QWord(y)))))
      },
      Seq(Seq("i like")).map { x =>
        TableRow(x.map(y => TableValue(y.split(" ").map(QWord(_)))))
      })
    val startingQuery = QueryLanguage.parse("(?<c1> {I, like, hate, it}+)").get
    val suggestions =
      QuerySuggester.suggestQuery(Seq(searcher), startingQuery, Map("test" -> table), Map(), ss,
        "test", true, SuggestQueryConfig(5, 2, 100, 1, -5, 0)).suggestions
    val bestScore = suggestions.head.score
    val bestSuggestions = suggestions.filter(_.score == bestScore)
    assert(bestSuggestions.map(_.query).toSet contains
      QueryLanguage.parse("(?<c1> {I, like, hate,it})").get)
  }

  it should "suggest changing a plus operator" in {
    val table = Table("test", Seq("c1"),
      Seq(Seq("not")).map { x =>
        TableRow(x.map(y => TableValue(Seq(QWord(y)))))
      },
      Seq(Seq("like"), Seq("hate")).map { x =>
        TableRow(x.map(y => TableValue(Seq(QWord(y)))))
      })
    val startingQuery = QueryLanguage.parse("(?<c1> .+) {mango, those, great}").get
    val suggestions =
      QuerySuggester.suggestQuery(Seq(searcher), startingQuery, Map("test" -> table), Map(), ss,
        "test", true, SuggestQueryConfig(5, 1, 100, 1, -5, -5))
    val bestScore = suggestions.suggestions.head.score
    // Fuzzy match to 1 to account for size penalties and the like
    val bestSuggestions = suggestions.suggestions.filter(_.score == bestScore).map(_.query)
    assert(bestSuggestions contains
      QueryLanguage.parse("(?<c1> RB+) {mango, those, great}").get)
  }

  it should "suggest altering repeated op" in {
    val table = Table("test", Seq("c1"),
      Seq(Seq("I like mango"), Seq("I hate those"), Seq("It tastes great")).map { x =>
        TableRow(x.map(y => TableValue(y.split(" ").map(word => QWord(word)))))
      },
      Seq())
    val startingQuery = QueryLanguage.parse("(.[1,3])").get
    val suggestions =
      QuerySuggester.suggestQuery(Seq(searcher), startingQuery, Map("test" -> table), Map(), ss,
        "test", true, SuggestQueryConfig(10, 2, 100, 100, -1, -1)).suggestions
    val bestScore = suggestions(0).score
    val bestQueries = suggestions.filter(_.score == bestScore)
    val queries = bestQueries.map(x => x.query)
    assert(queries contains QUnnamed(QSeq(Seq(QPos("PRP"), QPos("VBP"),
      QRepetition(QWildcard(), 0, 1)))))
  }

  it should "suggest removing disjunctions" in {
    val table = Table("test", Seq("c1"),
      Seq(Seq("I like")).map { x =>
        TableRow(x.map(y => TableValue(y.split(" ").map(word => QWord(word)))))
      },
      Seq(Seq("I hate")).map { x =>
        TableRow(x.map(y => TableValue(y.split(" ").map(word => QWord(word)))))
      })
    val startingQuery = QueryLanguage.parse("PRP {tastes, like, hate}").get
    val suggestions =
      QuerySuggester.suggestQuery(Seq(searcher), startingQuery, Map("test" -> table), Map(), ss,
        "test", true, SuggestQueryConfig(10, 2, 100, 100, -1, -1))
    val orig = suggestions.original
    assertResult((1, 1, 1))((orig.positiveScore, orig.negativeScore, orig.unlabelledScore))
    val bestScore = suggestions.suggestions.head.score
    val bestQueries = suggestions.suggestions.filter(_.score == bestScore)
    val best = bestQueries.head
    assertResult((1, 0, 0))((best.positiveScore, best.negativeScore, best.unlabelledScore))
    val queries = bestQueries.map(x => x.query)
    assert(queries contains QueryLanguage.parse("(PRP like)").get)
  }
}
