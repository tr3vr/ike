package org.allenai.ike.index

import org.allenai.datastore.Datastore
import org.allenai.nlpstack.core.{ ChunkedToken, Lemmatized }

import nl.inl.blacklab.index.Indexer

import java.io.{ File, StringReader }
import java.net.URI
import java.nio.file.{ Files, Paths }

object CreateIndex extends App {
  def addTo(indexer: Indexer)(text: IndexableText): Unit = {
    val xml = XmlSerialization.xml(text)
    val id = text.idText.id
    indexer.index(id, new StringReader(xml.toString()))
  }

  case class Options(
    destinationDir: File = null,
    batchSize: Int = 1000,
    textSource: URI = null,
    oneSentPerDoc: Boolean = true
  )

  val parser = new scopt.OptionParser[Options](this.getClass.getSimpleName.stripSuffix("$")) {
    opt[File]('d', "destination") required () action { (d, o) =>
      o.copy(destinationDir = d)
    } text "Directory to create the index in"

    opt[Int]('b', "batchSize") action { (b, o) =>
      o.copy(batchSize = b)
    } text "Batch size"

    opt[URI]('t', "textSource") required () action { (t, o) =>
      o.copy(textSource = t)
    } text "URL of a file or directory to load the text from"

    opt[Unit]('o', "oneSentencePerDoc") action { (_, o) =>
      o.copy(oneSentPerDoc = true)
    }
    help("help")
  }

  def getIdTextsForTextSource(textSource: URI): Iterator[IdText] = {
    textSource.getScheme match {
      case "file" =>
        val path = Paths.get(textSource)
        if (Files.isDirectory(path)) {
          IdText.fromDirectory(path.toFile)
        } else {
          IdText.fromFlatFile(path.toFile)
        }
      case "datastore" =>
        val locator = Datastore.locatorFromUrl(textSource)
        if (locator.directory) {
          IdText.fromDirectory(locator.path.toFile)
        } else {
          IdText.fromFlatFile(locator.path.toFile)
        }
      case otherAuthority =>
        throw new RuntimeException(s"URL scheme not supported: $otherAuthority")
    }
  }

  def process(idText: IdText, oneSentPerDoc: Boolean): Seq[IndexableText] = {

    def indexableToken(lemmatized: Lemmatized[ChunkedToken]): IndexableToken = {
      val word = lemmatized.token.string
      val pos = lemmatized.token.postag
      val lemma = lemmatized.lemma
      val chunk = lemmatized.token.chunk
      IndexableToken(word, pos, lemma, chunk)
    }

    if (oneSentPerDoc) {
      val sents: Seq[Seq[Lemmatized[ChunkedToken]]] = NlpAnnotate.annotate(idText.text)
      sents.zipWithIndex.filter(_._1.nonEmpty).map {
        case (sent, index) =>
          val text = idText.text.substring(
            sent.head.token.offset,
            sent.last.token.offset + sent.last.token.string.length
          )
          val sentenceIdText = IdText(s"${idText.id}-$index", text)

          IndexableText(sentenceIdText, Seq(sent map indexableToken))
      }
    } else {
      val text = idText.text
      val sents: Seq[Seq[IndexableToken]] = for {
        sent <- NlpAnnotate.annotate(text)
        indexableSent = sent map indexableToken
      } yield indexableSent
      Seq(IndexableText(idText, sents))
    }
  }

  def processBatch(batch: Seq[IdText], oneSentPerDoc: Boolean): Seq[IndexableText] =
    batch.toArray.par.map(idText => process(idText, oneSentPerDoc)).flatten.seq

  parser.parse(args, Options()) foreach { options =>
    val indexDir = options.destinationDir
    val batchSize = options.batchSize
    val idTexts: Iterator[IdText] = getIdTextsForTextSource(options.textSource)

    val indexer = new Indexer(indexDir, true, classOf[AnnotationIndexer])
    val indexableTexts = for {
      batch <- idTexts.grouped(batchSize)
      batchResults = processBatch(batch, options.oneSentPerDoc)
      result <- batchResults
    } yield result

    indexableTexts foreach addTo(indexer)
    indexer.close()
  }
}
