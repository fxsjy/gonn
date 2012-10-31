package gonn

import (
	"math"
	"fmt"
	"math/rand"
	"time"
)

type RBFNetwork struct{
	mInputCount int
	mInputLayer []float64
	mOutputLayer []float64
	mCenters [][]float64
	mWeightOutput [][]float64
	mLastChangeOutput [][]float64
	mRegression bool
	mRate1 float64
	mRate2 float64
}

func DefaultRBFNetwork(iInputCount,iOutputCount,iCenters int,iRegression bool) *RBFNetwork{
	return NewRBFNetwork(iInputCount,iOutputCount,iCenters,iRegression,0.25,0.1)
}


func NewRBFNetwork(iInputCount,iOutputCount,iCenters int,iRegression bool,iRate1,iRate2 float64) * RBFNetwork{
	self := &RBFNetwork{}
	self.mInputCount = iInputCount
	rand.Seed(time.Now().UnixNano())
	self.mInputLayer = make([]float64,iCenters+1)
	self.mOutputLayer = make([]float64,iOutputCount)
	self.mCenters = make([][]float64,iCenters)
	self.mWeightOutput = randomMatrix(iOutputCount, iCenters+1, -1.0, 1.0)
	self.mLastChangeOutput = makeMatrix(iOutputCount, iCenters+1, 0.0)
	self.mRegression = iRegression
	self.mRate1 = iRate1
	self.mRate2 = iRate2
	return self
}

func (self *RBFNetwork) Forward(input []float64) []float64{
	return self.ForwardRBF(self.make_rbf(input))
}

func (self *RBFNetwork) ForwardRBF(input []float64) []float64{
	if len(input)+1 != len(self.mInputLayer) {
		panic("amount of input variable doesn't match")
	}
	for i := 0; i < len(input); i++ {
		self.mInputLayer[i] = input[i]
	}
	self.mInputLayer[len(self.mInputLayer)-1] = 1.0 //bias node for input layer

	for i := 0; i < len(self.mOutputLayer); i++ {
		sum := 0.0
		for j := 0; j < len(self.mInputLayer); j++ {
			sum += self.mInputLayer[j] * self.mWeightOutput[i][j]
		}
		if(self.mRegression){
			self.mOutputLayer[i] = sum
		}else{
			self.mOutputLayer[i] = sigmoid(sum)
		}
	}

	return self.mOutputLayer[:]
}


func (self *RBFNetwork) Feedback(target []float64) {
	for i := 0; i < len(self.mOutputLayer); i++ {
		err_i := self.mOutputLayer[i] - target[i]
		for j := 0; j < len(self.mInputLayer); j++ {
			change := 0.0
			delta := 0.0
			if self.mRegression {
				delta = err_i
			} else {
				delta = err_i * dsigmoid(self.mOutputLayer[i])
			}
			change = self.mRate1*delta*self.mInputLayer[j] + self.mRate2*self.mLastChangeOutput[i][j]
			self.mWeightOutput[i][j] -= change
			self.mLastChangeOutput[i][j] = change
		}
	}
}


func (self *RBFNetwork) CalcError(target []float64) float64 {
	errSum := 0.0
	for i := 0; i < len(self.mOutputLayer); i++ {
		err := self.mOutputLayer[i] - target[i]
		errSum += 0.5 * err * err
	}
	return errSum
}

func (self *RBFNetwork) make_rbf(input []float64) []float64{
	result := make([]float64,len(self.mCenters))
	div := 0.0
	for j:=0;j<len(self.mCenters);j++{
		sum := 0.0
		for i:=0;i<self.mInputCount;i++{
			delta := input[i] - self.mCenters[j][i]
			sum += delta*delta
		}
		result[j] = math.Exp(-8*sum)
		div += result[j]
	}
	for j:=0;j<len(self.mCenters);j++{
		result[j] = result[j] / div
	}
	return result
}


func (self *RBFNetwork) Train(inputs [][]float64, targets [][]float64, iteration int) {
	if len(inputs[0]) != self.mInputCount {
		panic("amount of input variable doesn't match")
	}
	if len(targets[0]) != len(self.mOutputLayer) {
		panic("amount of output variable doesn't match")
	}
	if len(self.mCenters)>len(inputs){
		panic("too many centers, should be less than samples count")
	}
	sf_idx := genRandomIdx(len(inputs))
	for i:=0;i<len(self.mCenters);i++{
		self.mCenters[i] = inputs[sf_idx[i]] //random centers
	}

	r_inputs := make([][]float64,len(inputs))
	for i:=0;i<len(r_inputs);i++{
		r_inputs[i] = self.make_rbf(inputs[i])
		if (i+1)%10 ==0{
			fmt.Printf("genrate the %vth rbf vector\r",i+1)
		}
	}
	fmt.Println("")

	iter_flag := -1
	for i := 0; i < iteration; i++ {
		idx_ary := genRandomIdx(len(inputs))
		cur_err := 0.0
		for j := 0; j < len(inputs); j++ {
			self.ForwardRBF(r_inputs[idx_ary[j]])
			self.Feedback(targets[idx_ary[j]])
			cur_err += self.CalcError(targets[idx_ary[j]])
			if (j+1)%1000 == 0 {
				if iter_flag != i {
					fmt.Println("")
					iter_flag = i
				}
				fmt.Printf("iteration %v / progress %.2f %% \r", i+1, float64(j)*100/float64(len(inputs)))
			}
		}
		if (iteration >= 10 && (i+1)%(iteration/10) == 0) || iteration < 10 {
			fmt.Printf("\niteration %v err: %.5f", i+1, cur_err / float64(len(inputs)))
		}
	}
	fmt.Println("\ndone.")
}
