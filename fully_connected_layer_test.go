package cnns

import (
	"testing"

	"github.com/LdDl/cnns/tensor"
)

func TestFullyConectedSize(t *testing.T) {
	fc := NewFullyConnectedLayer(&tensor.TDsize{X: 6, Y: 7, Z: 1}, 3)
	correct := tensor.TDsize{X: 3, Y: 1, Z: 1}
	outSize := fc.GetOutputSize()
	if outSize.X != correct.X {
		t.Errorf("X dimension should be of value %d, but got %d", correct.X, outSize.X)
	}
	if outSize.Y != correct.Y {
		t.Errorf("Y dimension should be of value %d, but got %d", correct.Y, outSize.Y)
	}
	if outSize.Z != correct.Z {
		t.Errorf("Z dimension should be of value %d, but got %d", correct.Z, outSize.Z)
	}
}
