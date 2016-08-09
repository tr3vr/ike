package org.allenai.ike

import org.allenai.common.Config._
import org.allenai.common.{ Resource, Logging }
import org.allenai.ike.index.{ AnnotationIndexer, CreateIndex, IdText }
import org.allenai.ike.patterns.NamedPattern
import org.allenai.ike.persistence.Tablestore

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.typesafe.config.{ Config, ConfigFactory }
import nl.inl.blacklab.index.Indexer
import nl.inl.blacklab.search.Searcher
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.rdd.RDD
import scopt.OptionParser

import java.io.{ File, PrintWriter }
import java.nio.file.{ Files, Paths }

import scala.collection.JavaConverters._
import scala.util.{ Failure, Success }

/** A CLI that can run specified patterns over selected corpus indices
  * and dumps a table corresponding to each specified pattern to the
  * user-specified directory path. Tables will be in `tsv` format, which
  * is the same format as when we download tables via the IKE UI (Tables
  * tab).
  * The indices are specified in the config file (`application.conf`).
  */
object IkeBatchSearch extends App with Logging {
  val WINDOW_SIZE = 10000

  val mapper = new ObjectMapper().registerModule(DefaultScalaModule)

  /** Arguments to run IkeBatchSearch */
  case class IkeBatchSearchOptions(
    // Path to directory with text corpora to search over. If not provided,
    // the patterns will be searched over the indices listed in the application config.
    // TODO: Re-enable use of config sources.
    textSource: String = "",

    // TODO: Document me.
    indexPath: Option[String] = None,

    // input file containing an IKE pattern name and a corresponding
    // output table name per line in the following format:
    // <patternName>\t<tableName>
    // <patternName> is expected to refer to one of the named patterns the user
    // with specified account (userEmail) has saved via the IKE web tool.
    patternFile: String = "",

    // account info to get named patterns from. These patterns might contain
    // references to tables saved in the user's account. Currently IKE does
    // not enforce any security around user data (tables and patterns), so
    // we can look up any user's data. Eventually we should add permissions on
    // tables/patterns and required authentication.
    userEmail: String = "",
    // path to output directory with generated tables
    outputTableDir: String = ""
  )

  val optionParser: OptionParser[IkeBatchSearchOptions] = new scopt.OptionParser[IkeBatchSearchOptions]("") {
    // TODO: Add Kryo serialization for URIs and change this to one.
    opt[String]("text-source") action { (textSourceVal, options) =>
      options.copy(textSource = textSourceVal)
    } text ("URL of a file or directory to load the text from. This can be a local file (file:// " +
        "or absolute path) or an S3 URL (s3://).")
    opt[String]("index-path") action { (indexPathVal, options) =>
      options.copy(indexPath = Some(indexPathVal))
    } text ("TODO: Document me: location of index to reuse or index to build.")
    // TODO: Same here!
    opt[String]("pattern-file") valueName ("<file>") required () action {
       (patternFileVal, options) =>
         options.copy(patternFile = patternFileVal)
    } text ("Input pattern file")
    opt[String]("user-email") valueName ("<string>") required () action { (userEmailVal, options) =>
      options.copy(userEmail = userEmailVal)
    } text ("IKE User account email")
    opt[String]("output-dir") valueName ("<string>") required () action { (outputDirVal, options) =>
      options.copy(outputTableDir = outputDirVal)
    } text ("Directory path for output tables")
    help("help") text ("Prints this help.")
  }

  optionParser.parse(args, IkeBatchSearchOptions()) foreach { batchSearchOptions =>
    // Pull up the tables and patterns saved for the provided user account.
    val (tables, patterns) = (
      Tablestore.tables(batchSearchOptions.userEmail),
      Tablestore.namedPatterns(batchSearchOptions.userEmail)
    )

    val appConfig = ConfigFactory.load

    // Create a Searcher per index so that search can be parallelized.
    val searchAppConfig = appConfig.getConfigList("IkeToolWebapp.indices").asScala.map { config =>
      (config.getString("name"), DataFile.fromConfig(config).getAbsolutePath)
    }

    // Create instance of similar phrase searcher necessary to process queries such as
    // `cat ~ 10` which should return the 10 closest word2vec neighbors to `cat`.
    val word2vecPhrasesSearcher =
      new EmbeddingBasedPhrasesSearcher(appConfig[Config]("word2vecPhrasesSearcher"))

    // Create a table expander. This enables processing queries involving word2vec neighbors
    // for a specified table column's contents. For e.g., if we have a MaterialConductivity
    // table with an 'Energy' column that has some seed entries like "light" and "heat" and
    // we would like to find similar words like "thermal energy" and "electricity", the
    // `tableExpander` will compute the word2vec centroid of the specified column's current
    // entries and pull up n closest neighbors. The query that would trigger this will look
    // like: `$MaterialConductivity.Energy ~ 10`.
    val tableExpander = new SimilarPhrasesBasedTableExpander(word2vecPhrasesSearcher)

    /** Search for a pattern with a pre-created @Searcher and save the results to the given file.
      * Results are saved as JSON-serialized @BlackLabResult objects.
      * @param searcher the searcher to query with
      * @param pattern the pattern to search
      * @param outputFile the file results should be saved in
      */
    def searchForPattern(searcher: Searcher, pattern: NamedPattern, outputFile: File): Unit = {
      SearchApp.parse(Left(pattern.pattern)) match {
        case Success(query) =>
          // Get a simplified (ready-to-execute) version of the query.
          val interpolatedQuery = QueryLanguage.interpolateQuery(
            query,
            tables,
            patterns,
            word2vecPhrasesSearcher,
            tableExpander
          ).get

          val results = SearchApp.search(
            interpolatedQuery,
            searcher,
            batchSearchOptions.textSource,
            Some(WINDOW_SIZE))
          Resource.using(new PrintWriter(outputFile)) { writer =>
            results.foreach { result =>
              writer.println(mapper.writeValueAsString(result))
            }
          }

        case Failure(f) =>
          logger.error(s"Unable to parse pattern  ${pattern.name}", f)
      }
    }

    /** Split a pattern line into a pattern name and table name, as a tuple
      *
      * @param patternLine the pattern line to split
      * @return a tuple with (pattern name, table name)
      */
    def splitPattern(patternLine: String): Option[(String, String)] = {
      patternLine.split("\t").map(_.trim) match {
        case Array(patternName, tableName, _*) => Some((patternName, tableName))
        case _ => None
      }
    }

    val sparkConf = new SparkConf()
      .setAppName(this.getClass.getSimpleName)
      .setMaster("local[*]")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.kryo.registrator", "org.allenai.ike.IkeKryoRegistrator")
    val sparkContext = new SparkContext(sparkConf)

    val patternLines = sparkContext.textFile(batchSearchOptions.patternFile).collect

    // Open the index directory or create a temp directory if one hasn't been provided.
    val indexParentDir = batchSearchOptions.indexPath.map { path =>
      val dir = new File(path)
      dir.mkdir()
      dir
    }.getOrElse {
      val tempDir = Files.createTempDirectory("ike").toFile
      tempDir.deleteOnExit()
      tempDir
    }

    // Find all existing index directories. If there are none, we'll need to build an index.
    // TODO: Resuming a partial index would be a nice feature.
    val indexDirectories = indexParentDir.listFiles().filter(_.isDirectory)
    val indexPaths = if (indexDirectories.nonEmpty) {
      indexDirectories.map(_.getAbsolutePath)
    } else {
      // TODO: This partition count is totally arbitrary.
      val textLines = sparkContext.textFile(batchSearchOptions.textSource).repartition(16)

      // Create an index for each partition.
      val indexParentPath = indexParentDir.getAbsolutePath
      textLines.mapPartitionsWithIndex {
        case (partIndex, partition) =>
          val indexDir = Files.createDirectory(Paths.get(indexParentPath + "/" + partIndex)).toFile
          val indexer = new Indexer(indexDir, true, classOf[AnnotationIndexer])
          partition.foreach { sentence =>
            val idText = IdText(batchSearchOptions.textSource, sentence)
            val processedText = CreateIndex.process(idText, oneSentPerDoc = true)
            if (processedText.nonEmpty) {
              CreateIndex.addTo(indexer)(processedText.head)
            }
          }
          indexer.close()

          Seq(indexDir.getAbsolutePath).toIterator
      }.collect
    }

    // For each index and pattern, search and dump the serialized results.
    val patternResults: Seq[(String, String)] = indexPaths.toSeq.par.flatMap { indexPath =>
      // Create a searcher for this index.
      val indexDir = new File(indexPath)
      val searcher = Searcher.open(indexDir)

      // Create a temporary output file for this (pattern, index) pair.
      val tempDir = Files.createTempDirectory("ike").toFile
      tempDir.deleteOnExit()

      // TODO: Emit an error if any two patterns use the same table name.
      patternLines.map { patternLine =>
        // Process the pattern and run the query.
        splitPattern(patternLine) match {
          case Some((patternName, tableName)) =>
            logger.info(s"Processing pattern $patternName to produce table $tableName")
            patterns.get(patternName) match {
              case Some(pattern) =>
                val tempFile = new File(tempDir, tableName + ".json")
                searchForPattern(searcher, pattern, tempFile)
                (tableName, tempFile.getAbsolutePath)
              case None =>
                logger.error(s"Pattern $patternName not found!")
                ("", "")
            }
          case _ =>
            logger.error(s"Invalid format in pattern file, line: $patternLine")
            ("", "")
        }
      }
    }.seq

    // Group all result files by pattern so we can dump them into a single RDD.
    val resultFilesByPattern = patternResults.groupBy {
      case (pattern, path) => pattern
    } map {
      case (pattern, pairs) => (pattern, pairs.unzip._2)
    }

    // Format the results and write to file.
    resultFilesByPattern.foreach {
      case (tableName, paths) =>
        formatTablesFromResults(sparkContext, tableName, batchSearchOptions.outputTableDir, paths)
    }

    println("Done")
    sys.exit(0)
  }

  def formatTablesFromResults(
    sparkContext: SparkContext,
    tableName: String,
    outputDirPath: String,
    resultPaths: Seq[String]
  ): Unit = {
    val blackLabResults: RDD[BlackLabResult] = resultPaths.map { path =>
      sparkContext.textFile(path).map {
        line => mapper.readValue(line, classOf[BlackLabResult])
      }
    }.reduce(_ ++ _)

    // Group results by their uniqued tabular form (TSV).
    val keyedResults = blackLabResults.groupBy { result =>
      val captureGroups = if (result.captureGroups.nonEmpty) {
        result.captureGroups
      } else {
        Map("Match" -> result.matchOffset)
      }
      val phrases = captureGroups.map {
        case (_, offset) =>
          val words = result.wordData.slice(offset.start, offset.end)
          words.map(_.word.toLowerCase.trim).mkString(" ")
      }
      phrases.mkString("\t")
    }

    // Generate column headers.
    val columnCount = keyedResults.take(1) match {
      case Array((_, results)) if results.nonEmpty =>
        results.head.captureGroups.size
      case _ => 0
    }
    val columnHeaders = SearchResultGrouper.generateColumnNamesForNewTable(columnCount)

    val hitsWithCounts = keyedResults.map {
      case (key, value) => (key, value.size)
    }.sortBy(-_._2)

    // Output the results for WebUI import to a flat file.
    val webFile = new File(outputDirPath, tableName + "_webui.tsv")
    Resource.using(new PrintWriter(webFile)) { writer =>
      writer.println(columnHeaders.mkString("\t") + "\tlabel\tprovenance")
      hitsWithCounts.toLocalIterator.foreach {
        case (output, count) =>
          // TODO: Create an 'unlabeled' category for the batch mode scenario.
          // TODO: Save provenance.
          writer.println(output + "\tpositive")
      }
    }

    // Output the results for human readability.
    val prettyFile = new File(outputDirPath, tableName + "_pretty.tsv")
    Resource.using(new PrintWriter(prettyFile)) { writer =>
      writer.println(columnHeaders.mkString("\t") + "\tHitCounts")
      hitsWithCounts.toLocalIterator.foreach {
        case (output, count) => writer.println(output + "\t" + count)
      }
    }
  }
}
