package neuralnet

import neuralnet.NeuralNet.{ALayer, FCLayer, LayerType}
import chisel3._

class LayerFactory {

  def apply(layerType: LayerType): Layer = {
     Module(layerType match {
      case FCLayer(params) => new FullyConnectedLayer(params)
      case ALayer(params) => new ActivationLayer(params)
    })
  }
}
