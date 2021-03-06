package tensor

import (
	"fmt"
)

// Tensor - Structure for storing data of float64. Actually we can't call this Tensor in terms of math: this one just implements 3 dimensions
// Data - one-dimensional array of float64;
// Size - tensor's data size (see "TDsize" structure).
type Tensor struct {
	Data []float64
	Size *TDsize
}

// NewTensor - Constructor for Tensor type.
/*
	x - number of columns (width);
	y - number of rows (height);
	z - depth.
*/
func NewTensor(x, y, z int) *Tensor {
	return &Tensor{
		Data: make([]float64, x*y*z),
		Size: &TDsize{
			X: x,
			Y: y,
			Z: z,
		},
	}
}

// NewTensorCopy - Constructor for Tensor type.
/*
	t - *Tensor which you want to copy.
*/
func NewTensorCopy(t *Tensor) *Tensor {
	return &Tensor{
		Data: t.Data,
		Size: &TDsize{
			X: t.Size.X,
			Y: t.Size.Y,
			Z: t.Size.Z,
		},
	}
}

// Get - Return [i][j][k]-th element.
/*
	x - row;
	y - col;
	z - depth.
*/
func (t *Tensor) Get(x, y, z int) float64 {
	return t.Data[z*t.Size.X*t.Size.Y+y*t.Size.X+x]
}

// Set - Set [i][j][k]-th element with value.
/*
	x - row;
	y - col;
	z - depth;
	value - value of float64.
*/
func (t *Tensor) Set(x, y, z int, val float64) {
	t.Data[z*t.Size.X*t.Size.Y+y*t.Size.X+x] = val
}

// SetAdd - Add value to [i][j][k]-th element
/*
	x - row;
	y - col;
	z - depth;
	value - value of float64.
*/
func (t *Tensor) SetAdd(x, y, z int, val float64) {
	t.Data[z*t.Size.X*t.Size.Y+y*t.Size.X+x] += val
}

// SetData3D - Set data for *Tensor (as 3-d array)
/*
	data - 3-D array of float64.
*/
func (t *Tensor) SetData3D(data [][][]float64) {
	z := len(data)       // depth
	y := len(data[0])    // height (number of rows)
	x := len(data[0][0]) // width (number of columns)
	for i := 0; i < x; i++ {
		for j := 0; j < y; j++ {
			for k := 0; k < z; k++ {
				t.Set(i, j, k, data[k][j][i])
			}
		}
	}
}

// SetData - Set data for *Tensor
/*
	r - number of rows;
	c - number of columns (width);
	d - depth;
	data - 1-D array of float64.
*/
func (t *Tensor) SetData(c, r, d int, data []float64) {
	for i := 0; i < c; i++ {
		for j := 0; j < r; j++ {
			for k := 0; k < d; k++ {
				t.Set(i, j, k, data[k*c*d+j*c+i])
			}
		}
	}
}

// Print - Pretty print for *Tensor (10 decimal places)
func (t *Tensor) Print() {
	mx := t.Size.X
	my := t.Size.Y
	mz := t.Size.Z
	for z := 0; z < mz; z++ {
		fmt.Printf("Dim: %v\n", z)
		for y := 0; y < my; y++ {
			for x := 0; x < mx; x++ {
				fmt.Printf("%.15f\t", t.Get(x, y, z))
			}
			fmt.Println()
		}
	}
}

// GetData3D - Return *Tensor as 3-D array
func (t *Tensor) GetData3D() [][][]float64 {
	mx := t.Size.X
	my := t.Size.Y
	mz := t.Size.Z
	ret := make([][][]float64, mz)
	for z := 0; z < mz; z++ {
		ret[z] = make([][]float64, my)
		for y := 0; y < my; y++ {
			ret[z][y] = make([]float64, mx)
			for x := 0; x < mx; x++ {
				ret[z][y][x] = t.Get(x, y, z)
			}
		}
	}
	return ret
}

// IsEqualDims Returns true if dimensions of two tensors are equal, otherwise -> false.
func (t *Tensor) IsEqualDims(t2 *Tensor) bool {
	if t.Size.X != t2.Size.X || t.Size.Y != t2.Size.Y || t.Size.Z != t2.Size.Z {
		return false
	}
	return true
}
