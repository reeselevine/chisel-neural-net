package neuralnet

import chisel3._
import chisel3.experimental.{ChiselEnum, FixedPoint}
import chisel3.util._
import neuralnet.NeuralNet._

//todo: parameterize some of these things
object NeuralNet {
  val DataWidth = 32.W
  val DataBinaryPoint = 16.BP
  val LearningRate = 0.1
  val MaxTrainingSamples = 1000
  val MaxPredictSamples = 10

  object NeuronState extends ChiselEnum {
    val ready, reset, forwardProp, backwardProp = Value
  }

  sealed trait LayerType {
    def params: LayerParams
  }
  case class FCLayer(params: FullyConnectedLayerParams) extends LayerType
  case class ALayer(params: LayerParams) extends LayerType
}

case class NeuralNetParams(
                            inputSize: Int,
                            outputSize: Int,
                            trainingEpochs: Int,
                            layers: Seq[LayerType])

/**
 * IO for the neural net. The sample is used as input both for training and predicting, while the validation
 * is only used for training. Result is only valid during predicting.
 */
class NeuralNetIO(params: NeuralNetParams) extends Bundle {
  val train = Input(Bool())
  val predict = Input(Bool())
  val sample = Input(Vec(params.inputSize, FixedPoint(DataWidth, DataBinaryPoint)))
  val validation = Input(Vec(params.inputSize, FixedPoint(DataWidth, DataBinaryPoint)))
  val numSamples = Flipped(Decoupled(UInt(32.W)))
  val result = Decoupled(Vec(params.outputSize, FixedPoint(DataWidth, DataBinaryPoint)))
}

class NeuralNet(
                 params: NeuralNetParams,
                 layerFactory: LayerFactory) extends Module {

  val ready :: writingTrainingData :: trainingForwardProp :: trainingBackwardProp :: predicting :: Nil = Enum(5)

  val io = IO(new NeuralNetIO(params))

  // Way to define a set of layers with registers to hold the output from each layer. Output from one layer is used
  // as input to next layer for forward propagation. Goal is to get this working, then think about optimization.
  val layersWithOutputRegs = params.layers.map { layer =>
    val initializedLayer = layerFactory(layer)
    initializedLayer.io.nextState.valid := false.B
    initializedLayer.io.nextState.bits := NeuronState.ready
    initializedLayer.io.input.valid := false.B
    initializedLayer.io.input.bits := VecInit(Seq.fill(layer.params.inputSize)(0.F(DataWidth, DataBinaryPoint)))
    initializedLayer.io.output.ready := false.B
    initializedLayer.io.input_error.ready := true.B
    initializedLayer.io.output_error.valid := false.B
    initializedLayer.io.output_error.bits := VecInit(Seq.fill(layer.params.outputSize)(0.F(DataWidth, DataBinaryPoint)))
    (initializedLayer, RegInit(VecInit(Seq.fill(params.outputSize)(0.F(DataWidth, DataBinaryPoint)))))
  }

  val lastOutput = layersWithOutputRegs.last._2

  /** Memory used to store the training data, so it can be run over multiple epochs easily. */
  val trainingSamples = Mem(MaxTrainingSamples, Vec(params.inputSize, FixedPoint(DataWidth, DataBinaryPoint)))
  val validationSet = Mem(MaxTrainingSamples, Vec(params.inputSize, FixedPoint(DataWidth, DataBinaryPoint)))

  val curEpoch = Counter(params.trainingEpochs)
  val numSamples = RegInit(0.U(32.W))
  val sampleIndex = RegInit(0.U(32.W))
  val state = RegInit(ready)
  val curLayer = Counter(layersWithOutputRegs.length)

  io.numSamples.ready := true.B
  io.result.valid := false.B
  io.result.bits := VecInit(Seq.fill(params.outputSize)(0.F(DataWidth, DataBinaryPoint)))

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
        curLayer.reset()
      }
    }
    // prepare training data and validation set
    is(writingTrainingData) {
      trainingSamples.write(sampleIndex, io.sample)
      validationSet.write(sampleIndex, io.validation)
      when(sampleIndex === (numSamples - 1.U)) {
        state := trainingForwardProp
        sampleIndex := 0.U
      } .otherwise {
        sampleIndex := sampleIndex + 1.U
      }
    }
    is(trainingForwardProp) {
      layersWithOutputRegs.zipWithIndex.foreach {
        case ((layer, outputReg), idx) =>
          when(curLayer.value === idx.U) {
            layer.io.output.ready := true.B
            layer.io.nextState.valid := true.B
            layer.io.nextState.bits := NeuronState.forwardProp
            layer.io.input.valid := true.B
            if (idx == 0) {
              layer.io.input.bits := trainingSamples(sampleIndex)
            } else {
              layer.io.input.bits := layersWithOutputRegs(idx - 1)._2
            }
          }
          when(layer.io.output.valid) {
            layer.io.nextState.valid := false.B
            outputReg := layer.io.output.bits
            when(curLayer.inc()) {
              state := trainingBackwardProp
              // loss function derivative, use mean squared error
              lastOutput := VecInit(Seq.tabulate(params.outputSize){ i =>
                (validationSet(sampleIndex)(i.U) - lastOutput(i.U)) * (2/params.outputSize).F(DataWidth, DataBinaryPoint)
              })
            }
          }
      }
    }
    is(trainingBackwardProp) {
      layersWithOutputRegs.zipWithIndex.reverse.foreach {
        case ((layer, outputReg), idx) =>
          when(curLayer.value === idx.U) {
            layer.io.input_error.ready := true.B
            layer.io.nextState.valid := true.B
            layer.io.nextState.bits := NeuronState.backwardProp
            layer.io.output_error.valid := true.B
            layer.io.output_error.bits := outputReg
            when(sampleIndex === numSamples - 1.U) {
              when(curEpoch.inc()) {
                state := ready
              } .otherwise {
                state := trainingForwardProp
              }
            } .otherwise {
              sampleIndex := sampleIndex + 1.U
              state := trainingForwardProp
            }
          }
      }
    }
    is(predicting) {
      layersWithOutputRegs.zipWithIndex.foreach {
        case ((layer, outputReg), idx) =>
          when(curLayer.value === idx.U) {
            layer.io.output.ready := true.B
            layer.io.nextState.valid := true.B
            layer.io.nextState.bits := NeuronState.forwardProp
            layer.io.input.valid := true.B
            if (idx == 0) {
              layer.io.input.bits := io.sample
            } else {
              layer.io.input.bits := layersWithOutputRegs(idx - 1)._2
            }
          }
          when(layer.io.output.valid) {
            layer.io.nextState.valid := false.B
            outputReg := layer.io.output.bits
            when(curLayer.inc()) {
              io.result.valid := true.B
              io.result.bits := layer.io.output.bits
              when(sampleIndex === numSamples - 1.U) {
                state := ready
              } .otherwise {
                sampleIndex := sampleIndex + 1.U
              }
            }
          }
      }
    }
  }
}
