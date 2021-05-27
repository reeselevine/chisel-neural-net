package neuralnet

import chisel3._
import chisel3.experimental.FixedPoint
import chisel3.util._
import neuralnet.NeuralNet._

//todo: parameterize some of these things
object NeuralNet {
  val InputSize = 2
  val OutputSize = 1
  val DataWidth = 32.W
  val DataBinaryPoint = 16.BP
  val LearningEpochs = 100
  val LearningRate = 0.1
  val MaxTrainingSamples = 1000
  val MaxPredictSamples = 10
}

/**
 * IO for the neural net. The sample is used as input both for training and predicting, while the validation
 * is only used for training. Result is only valid during predicting.
 */
class NeuralNetIO extends Bundle {
  val train = Input(Bool())
  val predict = Input(Bool())
  val sample = Input(Vec(InputSize, FixedPoint(DataWidth, DataBinaryPoint)))
  val validation = Input(Vec(OutputSize, FixedPoint(DataWidth, DataBinaryPoint)))
  val numSamples = Flipped(Decoupled(UInt(32.W)))
  val result = Decoupled(Vec(OutputSize, FixedPoint(DataWidth, DataBinaryPoint)))
}

class NeuralNet extends Module {

  val ready :: writingTrainingData :: training :: predicting :: Nil = Enum(3)

  /** Need to figure out best way to build a modular net. But to start, we can manually define which layers
   * are being used and connect them as part of this module.
   */
  val fcLayer = Module(new FullyConnectedLayer)

  val io = IO(new NeuralNetIO)

  /** Memory used to store the training data, so it can be run over multiple epochs easily. */
  val trainingSamples = Mem(MaxTrainingSamples, Vec(InputSize, FixedPoint(DataWidth, DataBinaryPoint)))
  val validationSet = Mem(MaxTrainingSamples, Vec(InputSize, FixedPoint(DataWidth, DataBinaryPoint)))

  val curEpoch = Counter(LearningEpochs)
  val numSamples = RegInit(0.U(32.W))
  val sampleIndex = RegInit(0.U(32.W))
  val state = RegInit(ready)

  io.numSamples.ready := true.B
  io.result.valid := false.B
  io.result.bits := VecInit(Seq.fill(OutputSize)(0.F(DataWidth, DataBinaryPoint)))

  switch(state) {
    // initial state, input decides whether to go to train or predict
    is(ready) {
      when(io.train && io.numSamples.fire) {
        state := writingTrainingData
        numSamples := io.numSamples.bits
        sampleIndex := 0.U
      } .elsewhen(io.predict && io.numSamples.fire) {
        state := predicting
        numSamples := io.numSamples.bits
        sampleIndex := 0.U
      }
    }
    // prepare training data and validation set
    is(writingTrainingData) {
      trainingSamples.write(sampleIndex, io.sample)
      validationSet.write(sampleIndex, io.validation)
      when(sampleIndex === (numSamples - 1.U)) {
        state := training
        sampleIndex := 0.U
      } .otherwise {
        sampleIndex := sampleIndex + 1.U
      }
    }
    is(training) {
      //todo: training step (do number of epochs, forward/back-prop/loss fn
    }
    is(predicting) {
      //todo: do forward prop, send result back as output
    }
  }


}
