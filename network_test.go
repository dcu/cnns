package cnns

import (
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/LdDl/cnns/tensor"
	"github.com/stretchr/testify/require"
)

func init() {
}

func TestWholeNetExportToFileConv(t *testing.T) {
	rand.Seed(1600790650)

	c := require.New(t)

	net := &WholeNet{}
	conv := NewConvLayer(1, 1, 1, tensor.TDsize{X: 8, Y: 1, Z: 1})
	net.Layers = append(net.Layers, conv)

	train := tensor.NewTensor(8, 1, 1)
	train.SetData3D([][][]float64{[][]float64{[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}}})
	net.FeedForward(train)

	desired := tensor.NewTensor(8, 1, 1)
	desired.SetData3D([][][]float64{[][]float64{[]float64{0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1}}})
	c.NoError(net.Backpropagate(desired))

	net.Layers[0].PrintWeights()

	input := tensor.NewTensor(8, 1, 1)
	input.SetData3D([][][]float64{[][]float64{[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}}})
	net.FeedForward(input)

	fmt.Printf("%#v\n", net.Layers[0].GetOutput().Data)

	c.Equal([]float64{0.0222207747462706, 0.0444415494925412, 0.0666623242388118, 0.0888830989850824, 0.111103873731353, 0.1333246484776236, 0.1555454232238942, 0.1777661979701648}, net.Layers[0].GetOutput().Data)

	tmpfile, err := ioutil.TempFile("", "test_cnns.*.json")
	c.NoError(err)

	defer func() {
		_ = os.Remove(tmpfile.Name())
	}()

	c.NoError(net.ExportToFile(tmpfile.Name()))

	{
		loadedNet := &WholeNet{}
		c.NoError(loadedNet.ImportFromFile(tmpfile.Name(), false))

		loadedNet.FeedForward(input)

		fmt.Printf("%#v\n", loadedNet.Layers[0].GetOutput().Data)

		c.Equal([]float64{0.0222207747462706, 0.0444415494925412, 0.0666623242388118, 0.0888830989850824, 0.111103873731353, 0.1333246484776236, 0.1555454232238942, 0.1777661979701648}, loadedNet.Layers[0].GetOutput().Data)
	}
}

func TestWholeNetExportToFileFC(t *testing.T) {
	rand.Seed(1600790650)

	c := require.New(t)

	net := &WholeNet{}
	fullyconnected := NewFullyConnectedLayer(&tensor.TDsize{X: 8, Y: 1, Z: 1}, 3)
	net.Layers = append(net.Layers, fullyconnected)

	train := tensor.NewTensor(8, 1, 1)
	train.SetData3D([][][]float64{[][]float64{[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}}})
	net.FeedForward(train)

	desired := tensor.NewTensor(3, 1, 1)
	desired.SetData3D([][][]float64{[][]float64{[]float64{0.8, 0.7, 0.6}}})
	c.NoError(net.Backpropagate(desired))

	net.Layers[0].PrintWeights()

	input := tensor.NewTensor(8, 1, 1)
	input.SetData3D([][][]float64{[][]float64{[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}}})
	net.FeedForward(input)

	fmt.Printf("%#v\n", net.Layers[0].GetOutput().Data)

	c.Equal([]float64{0.2200627324127083, 0.10514781377947911, -0.04190095644513875}, net.Layers[0].GetOutput().Data)

	tmpfile, err := ioutil.TempFile("", "test_cnns.*.json")
	c.NoError(err)

	defer func() {
		_ = os.Remove(tmpfile.Name())
	}()

	c.NoError(net.ExportToFile(tmpfile.Name()))

	{
		loadedNet := &WholeNet{}
		c.NoError(loadedNet.ImportFromFile(tmpfile.Name(), false))

		loadedNet.FeedForward(input)

		fmt.Printf("%#v\n", loadedNet.Layers[0].GetOutput().Data)

		c.Equal([]float64{0.2200627324127083, 0.10514781377947911, -0.04190095644513875}, loadedNet.Layers[0].GetOutput().Data)
	}
}

func TestWholeNetExportToFileMultiple(t *testing.T) {
	rand.Seed(1600790650)

	c := require.New(t)

	conv := NewConvLayer(1, 2, 2, tensor.TDsize{X: 8, Y: 3, Z: 1})
	relu := NewReLULayer(conv.GetOutputSize())
	maxpool := NewMaxPoolingLayer(2, 2, relu.GetOutputSize())
	fullyconnected1 := NewFullyConnectedLayer(maxpool.GetOutputSize(), 3)
	fullyconnected2 := NewFullyConnectedLayer(fullyconnected1.GetOutputSize(), 6)
	fullyconnected3 := NewFullyConnectedLayer(fullyconnected2.GetOutputSize(), 3)

	var net WholeNet
	net.Layers = append(net.Layers, conv)
	net.Layers = append(net.Layers, relu)
	net.Layers = append(net.Layers, maxpool)
	net.Layers = append(net.Layers, fullyconnected1)
	net.Layers = append(net.Layers, fullyconnected2)
	net.Layers = append(net.Layers, fullyconnected3)

	// train and check
	train := tensor.NewTensor(8, 3, 1)
	train.SetData3D([][][]float64{[][]float64{
		[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
		[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
		[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
	}})
	net.FeedForward(train)

	desired := tensor.NewTensor(3, 1, 1)
	desired.SetData3D([][][]float64{[][]float64{
		[]float64{0.99, 0.01, 0.01},
	}})
	c.NoError(net.Backpropagate(desired))

	net.Layers[5].PrintWeights()

	input := tensor.NewTensor(8, 3, 1)
	input.SetData3D([][][]float64{[][]float64{
		[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
		[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
		[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
	}})
	net.FeedForward(input)

	fmt.Printf("%#v\n", net.Layers[5].GetOutput().Data)

	c.Equal([]float64{-0.008558651784486664, 0.001610323003517105, -0.0013409866222555157}, net.GetOutput().Data)

	tmpfile, err := ioutil.TempFile("", "test_cnns.*.json")
	c.NoError(err)

	defer func() {
		_ = os.Remove(tmpfile.Name())
	}()

	c.NoError(net.ExportToFile(tmpfile.Name()))

	{
		rand.Seed(time.Now().UnixNano())

		loadedNet := &WholeNet{}
		c.NoError(loadedNet.ImportFromFile(tmpfile.Name(), false))

		loadedNet.FeedForward(input)

		fmt.Printf("%#v\n", loadedNet.Layers[5].GetOutput().Data)

		c.Equal([]float64{-0.008558651784486664, 0.001610323003517105, -0.0013409866222555157}, loadedNet.GetOutput().Data)
	}
}
