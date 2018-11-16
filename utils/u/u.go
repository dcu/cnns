package u

import (
	"errors"
	"math"
	"math/rand"
)

// AndINT - Logical AND for two inputs of type int.
func AndINT(x, y int) int {
	firstBool := false
	secondBool := false
	if x == 1 {
		firstBool = true
	}
	if y == 1 {
		secondBool = true
	}
	outputBool := (firstBool && secondBool)
	outputInt := 0
	if outputBool == true {
		outputInt = 1
	}
	return outputInt
}

// OrINT - Logical OR for two inputs of type int.
func OrINT(x, y int) int {
	firstBool := false
	secondBool := false
	if x == 1 {
		firstBool = true
	}
	if y == 1 {
		secondBool = true
	}
	outputBool := (firstBool || secondBool)
	outputInt := 0
	if outputBool == true {
		outputInt = 1
	}
	return outputInt
}

// XorINT - Logical XOR for two inputs of type int.
func XorINT(x, y int) int {
	firstBool := false
	secondBool := false
	if x == 1 {
		firstBool = true
	}
	if y == 1 {
		secondBool = true
	}
	outputBool := (firstBool != secondBool)
	outputInt := 0
	if outputBool == true {
		outputInt = 1
	}
	return outputInt
}

// SuffleSlice - Shuffle a slice in random way.
func SuffleSlice(data []interface{}) []interface{} {
	for i := range data {
		j := rand.Intn(i + 1)
		data[i], data[j] = data[j], data[i]
	}
	return data
}

// RandomInt - Get random integer.
func RandomInt(min, max int) int {
	return rand.Intn(max-min) + min
}

// NormalizeRange - Normalizing range.
func NormalizeRange(f float64, max int, limitMin bool) int {
	if f <= 0 {
		return 0
	}
	max--
	if f >= float64(max) {
		return max
	}
	if limitMin {
		return int(math.Ceil(f))
	}
	return int(math.Floor(f))
}

// Matrix2D @experiments
type Matrix2D [][]float64

func flatten(f Matrix2D) (r, c int, d []float64, err error) {
	r = len(f)
	if r == 0 {
		return 0, 0, nil, errors.New("No row")
	}
	c = len(f[0])
	d = make([]float64, 0, r*c)
	for _, row := range f {
		if len(row) != c {
			return 0, 0, nil, errors.New("Ragge input")
		}
		d = append(d, row...)
	}
	return r, c, d, nil
}

func unflatten(r, c int, d []float64) Matrix2D {
	m := make(Matrix2D, r)
	for i := 0; i < r; i++ {
		m[i] = d[i*c : (i+1)*c]
	}
	return m
}

// Round Round float64 to 0 decimal places
func Round(v float64) float64 {
	if v >= 0 {
		return math.Floor(v + 0.5)
	} else {
		return math.Ceil(v - 0.5)
	}
}

// RoundPlaces Round float64 to N decimal places
func RoundPlaces(v float64, places int) float64 {
	shift := math.Pow(10, float64(places))
	return Round(v*shift) / shift
}
