package neuralnet

import chisel3._
import chisel3.tester._
import org.scalatest.FreeSpec
import FullyConnectedLayerSpec._
import chisel3.util._
import neuralnet.NeuralNet._
import neuralnet.NeuralNetSpec.{TestLayerFactory, buildBasicNet, defaultNeuralNetParams}

class NeuralNetSpec extends FreeSpec with ChiselScalatestTester {

  "A neural net should initialize and predict multiple samples" in {
    test(buildBasicNet()) { dut =>
      dut.io.numSamples.ready.expect(true.B)
      dut.io.state.expect(ready)
      dut.io.numSamples.valid.poke(true.B)
      dut.io.numSamples.bits.poke(2.U)
      dut.io.sample.foreach(bit => bit.poke(1.F(DataWidth, DataBinaryPoint)))
      dut.io.predict.poke(true.B)
      dut.clock.step(1)
      dut.io.state.expect(predicting)
      dut.io.result.ready.poke(true.B)
      dut.clock.step(1)
      dut.io.state.expect(predicting)
      dut.io.result.valid.expect(true.B)
      dut.clock.step(1)
      dut.io.state.expect(predicting)
      dut.io.result.valid.expect(true.B)
      dut.clock.step(1)
      dut.io.state.expect(ready)
    }
  }

  "A neural net should train by performing forward and backpropagation" in {
    test(buildBasicNet(layerFactory = new TestLayerFactory)) { dut =>
      dut.io.numSamples.valid.poke(true.B)
      dut.io.numSamples.bits.poke(2.U)
      dut.io.sample.foreach(bit => bit.poke(1.F(DataWidth, DataBinaryPoint)))
      dut.io.validation.foreach(bit => bit.poke(1.F(DataWidth, DataBinaryPoint)))
      dut.io.train.poke(true.B)
      dut.clock.step(1)
      dut.io.train.poke(false.B)
      dut.io.state.expect(writingTrainingData)
      dut.clock.step(2)
      dut.io.state.expect(trainingForwardProp)
      dut.clock.step(2)
      dut.io.state.expect(trainingBackwardProp)
      dut.clock.step(2)
      dut.io.state.expect(trainingForwardProp)
      dut.clock.step(2)
      dut.io.state.expect(trainingBackwardProp)
      dut.clock.step(2)
      dut.io.epoch.expect(1.U)
      dut.io.state.expect(trainingForwardProp)
      dut.clock.step(8)
      dut.io.epoch.expect(0.U)
      dut.io.state.expect(ready)
    }
  }

  "A neural network should train and predict xor" in {
    val trainingData = Seq(Seq(0.0, 0.0), Seq(0.0, 1.0), Seq(1.0, 0.0), Seq(1.0, 1.0))
    val validationData = Seq(Seq(0.0), Seq(1.0), Seq(1.0), Seq(0.0))
    val predictionData = Seq(Seq(1.0, 1.0), Seq(1.0, 0.0))
    val fcLayer1 = FullyConnectedLayerParams(2, 3, 0.5)
    val fcLayer2 = FullyConnectedLayerParams(3, 1, 0.5)
    val params = defaultNeuralNetParams.copy(layers = Seq(FCLayer(fcLayer1), FCLayer(fcLayer2)))
    test(buildBasicNet(params = params)) { dut =>
      dut.io.numSamples.valid.poke(true.B)
      dut.io.numSamples.bits.poke(4.U)
      dut.io.train.poke(true.B)
      dut.io.state.expect(ready)
      dut.clock.step(1)
      dut.io.train.poke(false.B)
      dut.io.numSamples.valid.poke(false.B)
      trainingData.indices.foreach { i =>
        writeTrainingData(trainingData(i), validationData(i), dut)
        dut.io.state.expect(writingTrainingData)
        dut.clock.step(1)
      }
      dut.io.state.expect(trainingForwardProp)
      dut.io.layer.expect(0.U)
      dut.clock.step(4)
      dut.io.state.expect(trainingBackwardProp)
      dut.clock.step(4)
      dut.io.state.expect(trainingForwardProp)
      dut.clock.step(8*7) //do the rest of the training

      //Back to ready state, let's try predicting some values
      dut.io.state.expect(ready)
      dut.io.numSamples.valid.poke(true.B)
      dut.io.numSamples.bits.poke(2.U)
      dut.io.predict.poke(true.B)
      dut.clock.step(1)
      dut.io.state.expect(predicting)
      predictionData(0).indices.foreach { j =>
        dut.io.sample(j).poke(predictionData(0)(j).F(DataWidth, DataBinaryPoint))
      }
      dut.io.result.ready.poke(true.B)
      dut.clock.step(3)
      dut.io.result.valid.expect(true.B)
      dut.clock.step(1)
      predictionData(1).indices.foreach { j =>
        dut.io.sample(j).poke(predictionData(1)(j).F(DataWidth, DataBinaryPoint))
      }
      dut.clock.step(1)
      dut.io.result.valid.expect(true.B)
      dut.clock.step(1)
      dut.io.state.expect(ready)
    }
  }

  def writeTrainingData(train: Seq[Double], validate: Seq[Double], dut: NeuralNet) = {
    train.indices.foreach { i =>
      dut.io.sample(i).poke(train(i).F(DataWidth, DataBinaryPoint))
    }
    validate.indices.foreach { i =>
      dut.io.validation(i).poke(validate(i).F(DataWidth, DataBinaryPoint))
    }
  }
}

object NeuralNetSpec {

  val defaultNeuralNetParams = NeuralNetParams(
    inputSize = 2,
    outputSize = 1,
    trainingEpochs = 2,
    layers = Seq(FCLayer(fcLayerDefaultParams)))

  /** A dummy layer that implements the [[NeuronState]] state machine, but simply returns
   * its input as output, for use in testing [[NeuralNet]] functionality. Note that for
   * this class to work, the input size must be the same as the output size. */
  class DummyLayer(params: LayerParams) extends Layer(params) {

    io.input.ready := true.B
    io.nextState.ready := true.B
    io.output.valid := false.B
    io.output.bits := VecInit(Seq.fill(params.outputSize)(0.F(DataWidth, DataBinaryPoint)))
    io.output_error.ready := true.B
    io.input_error.valid := false.B
    io.input_error.bits := VecInit(Seq.fill(params.inputSize)(0.F(DataWidth, DataBinaryPoint)))

    val state = RegInit(NeuronState.ready)

    switch(state) {
      is(NeuronState.ready) {
        when(io.nextState.fire()) {
          state := io.nextState.bits
        }
      }
      is(NeuronState.forwardProp) {
        when(io.input.fire()) {
          io.output.valid := true.B
          (0 until params.outputSize).foreach { i =>
            io.output.bits(i) := io.input.bits(i)
          }
        }
        when(io.nextState.fire()) {
          state := io.nextState.bits
        }
      }
      is(NeuronState.backwardProp) {
        when(io.output_error.fire()) {
          io.input_error.valid := true.B
          (0 until params.inputSize).foreach { i =>
            io.input_error.bits(i.U) := io.output_error.bits(i.U)
          }
        }
        when(io.nextState.fire()) {
          state := io.nextState.bits
        }
      }
    }
  }

  class TestLayerFactory extends LayerFactory {
    override def apply(layerType: NeuralNet.LayerType): Layer = {
      Module(new DummyLayer(layerType.params))
    }
  }

  def buildBasicNet(
                     layerFactory: LayerFactory = new LayerFactory,
                     params: NeuralNetParams = defaultNeuralNetParams): NeuralNet = {
    new NeuralNet(params, layerFactory)
  }
}
