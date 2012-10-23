package gonn

import (
	"math"
	"math/rand"
	"fmt"
)

type NeuralNetwork struct{
	mHiddenLayer []*Neural
	mInputLayer  []*Neural
	mOutputLayer []*Neural
	mWeightHidden [][]float64
	mWeightOutput [][]float64
	mLastChangeHidden [][]float64
	mLastChangeOutput [][]float64
	mOutput []float64
	mForwardDone chan bool
	mFeedbackDone chan bool
	mRegression bool
	mRate1 float64 //learning rate
	mRate2 float64
}

func makeMatrix(rows,colums int, value float64) [][]float64{
	mat := make([][]float64,rows)
	for i:=0;i<rows;i++{
		mat[i] = make([]float64,colums)
		for j:=0;j<colums;j++{
			mat[i][j] = value
		}
	}
	return mat
}

func randomMatrix(rows,colums int, lower, upper float64) [][]float64{
	mat := make([][]float64,rows)
	for i:=0;i<rows;i++{
		mat[i] = make([]float64,colums)
		for j:=0;j<colums;j++{
			mat[i][j] = rand.Float64()*(upper-lower) + lower
		}
	}
	return mat
}

func DefaultNetwork(iInputCount,iHiddenCount,iOutputCount int, iRegression bool) (*NeuralNetwork) {
	return NewNetwork(iInputCount,iHiddenCount,iOutputCount,iRegression, 0.01,0.001)
}


func NewNetwork(iInputCount,iHiddenCount,iOutputCount int, iRegression bool,iRate1,iRate2 float64) (*NeuralNetwork){
	iInputCount +=1
	network := &NeuralNetwork{}
	network.mRegression = iRegression
	network.mOutput = make([]float64,iOutputCount)
	network.mForwardDone = make(chan bool)
	network.mFeedbackDone = make(chan bool)
	network.mInputLayer = make([]*Neural,iInputCount)
	network.mRate1 = iRate1
	network.mRate2 = iRate2
	for i:=0;i<iInputCount;i++{
		network.mInputLayer[i] = NewNeural(network,0,i,1)
	}
	network.mHiddenLayer = make([]*Neural,iHiddenCount)
	for i:=0;i<iHiddenCount;i++{
		network.mHiddenLayer[i] = NewNeural(network,1,i,iInputCount)
	}
	network.mOutputLayer = make([]*Neural,iOutputCount)
	for i:=0;i<iOutputCount;i++{
		network.mOutputLayer[i] = NewNeural(network,2,i,iHiddenCount)
	}

	network.mWeightHidden = randomMatrix(iInputCount,iHiddenCount,-0.2,0.2)
	network.mWeightOutput = randomMatrix(iHiddenCount,iOutputCount,-2.0,2.0)

	network.mLastChangeHidden = makeMatrix(iInputCount,iHiddenCount,0.0)
	network.mLastChangeOutput = makeMatrix(iHiddenCount,iOutputCount,0.0)

	return network
}

func (self * NeuralNetwork) Start(){//start all the neurals in the network
	for _,n := range self.mInputLayer{
		n.start(self.mRegression)
	}
	for _,n := range self.mHiddenLayer{
		n.start(self.mRegression)
	}
	for _,n := range self.mOutputLayer{
		n.start(self.mRegression)
	}
}

func (self * NeuralNetwork) Stop(){//start all the neurals in the network

	for _,n := range self.mInputLayer{
		close(n.mInputChan)
		close(n.mFeedbackChan)
	}
	for _,n := range self.mHiddenLayer{
		close(n.mInputChan)
		close(n.mFeedbackChan)
	}
	for _,n := range self.mOutputLayer{
		close(n.mInputChan)
		close(n.mFeedbackChan)
	}
	close(self.mForwardDone)
	close(self.mFeedbackDone)
}


func (self * NeuralNetwork) Forward(input []float64 ) (output []float64){
	if len(input)+1 != len(self.mInputLayer){
		panic("amount of input variable doesn't match")
	}
	go func(){
		for i:=0;i<len(self.mInputLayer)-1;i++{
			self.mInputLayer[i].mInputChan <- input[i]
		}
		self.mInputLayer[len(self.mInputLayer)-1].mInputChan  <- 1.0 //bias node
	}()
	for i:=0;i<len(self.mOutput);i++{
		<-self.mForwardDone
	}
	return self.mOutput[:]
}

func (self * NeuralNetwork) Feedback(target []float64) {
	go func(){
		for i:=0;i<len(self.mOutput);i++{
			self.mOutputLayer[i].mFeedbackChan <- target[i]
		}
	}()
	for i:=0;i<len(self.mHiddenLayer);i++{
		<- self.mFeedbackDone
	}

}

func (self * NeuralNetwork) CalcError( target []float64) float64{
	errSum := 0.0
	for i:=0;i<len(self.mOutput);i++{
		err := self.mOutput[i] - target[i]
		errSum += 0.5 * err * err
	}
	return errSum
}

func genRandomIdx(N int) []int{
	A := make([]int,N)
	for i:=0;i<N;i++{
		A[i]=i
	}
	//randomize
	for i:=0;i<N;i++{
		j := i+int(rand.Float64() * float64 (N-i))
		A[i],A[j] = A[j],A[i]
	}
	return A
}

func (self * NeuralNetwork) Train(inputs [][]float64, targets [][]float64, iteration int) {
	if len(inputs[0])+1 != len(self.mInputLayer){
		panic("amount of input variable doesn't match")
	}
	if len(targets[0]) != len(self.mOutputLayer){
		panic("amount of output variable doesn't match")
	}
	old_err1 := 1.0
	old_err2 := 2.0
	
	for i:=0;i<iteration;i++{
		idx_ary := genRandomIdx(len(inputs))
		for j:=0;j<len(inputs);j++{
			self.Forward(inputs[idx_ary[j]])
			self.Feedback(targets[idx_ary[j]])
		}
		if i%100==0 {
			last_target := targets[len(targets)-1]
			cur_err := self.CalcError(last_target)
			fmt.Println("err: ", cur_err)
			if (old_err2 - old_err1 < 0.001) && (old_err1 - cur_err  < 0.001){//early stop
				break
			}	
			old_err2 = old_err1
			old_err1 = cur_err
			
		}
	}
}


type Neural struct{
	mInputChan chan float64
	mFeedbackChan chan float64
	mInputCount int
	mLayer int
	mNo int
	mNetwork * NeuralNetwork
	mValue float64
}

func NewNeural(iNetwork *NeuralNetwork, iLayer, iNo , iInputCount int) (*Neural){
	nerual := &Neural{}
	nerual.mNetwork = iNetwork
	nerual.mInputCount = iInputCount
	nerual.mLayer = iLayer
	nerual.mInputChan = make(chan float64)
	nerual.mFeedbackChan = make(chan float64)
	nerual.mNo = iNo
	nerual.mValue = 0.0
	return nerual
}


func sigmoid(X float64) float64{
	return math.Tanh(X)
}

func dsigmoid(Y float64) float64{
	return 1-Y*Y
}

func (self *Neural) start(regression bool){
	go func(){//forward loop
		defer func(){recover()} ()
		for {
			sum := 0.0
			for i:=0;i<self.mInputCount;i++{
				value := <- self.mInputChan
				sum += value
			}
			if self.mLayer==0 {//input layer
				for i:=0;i<len(self.mNetwork.mHiddenLayer);i++{
					self.mNetwork.mHiddenLayer[i].mInputChan <- sum * self.mNetwork.mWeightHidden[self.mNo][i]
				}
			}else if self.mLayer==1 {//hidden layer
				sum = sigmoid(sum)
				for i:=0;i<len(self.mNetwork.mOutputLayer);i++{
					self.mNetwork.mOutputLayer[i].mInputChan <- sum * self.mNetwork.mWeightOutput[self.mNo][i]
				}
			}else {//output layer
				if !regression{
					sum = sigmoid(sum)
				}
				self.mNetwork.mOutput[self.mNo] = sum 
				self.mNetwork.mForwardDone <- true
			}
			self.mValue = sum
		}

	}()

	go func(){//feedback loop
		defer func(){recover()} ()
		for{
			if self.mLayer==0{ //input layer
				return
			} else if self.mLayer==1{ //hidden layer
				err :=0.0
				for i:=0;i<len(self.mNetwork.mOutput);i++{
					err += <- self.mFeedbackChan
				}
				for i:=0;i<self.mInputCount;i++{
					change := err * dsigmoid(self.mValue) * self.mNetwork.mInputLayer[i].mValue
					self.mNetwork.mWeightHidden[i][self.mNo] -= (self.mNetwork.mRate1*change + self.mNetwork.mRate2*self.mNetwork.mLastChangeHidden[i][self.mNo])
					self.mNetwork.mLastChangeHidden[i][self.mNo] = change
				}
				self.mNetwork.mFeedbackDone <- true
			} else{ //output layer 
				target := <- self.mFeedbackChan
				err := self.mValue - target
				for i:=0;i<self.mInputCount;i++{
					self.mNetwork.mHiddenLayer[i].mFeedbackChan <- err * self.mNetwork.mWeightOutput[i][self.mNo]
				}
				if regression{
					for i:=0;i<self.mInputCount;i++{
						change := err * self.mNetwork.mHiddenLayer[i].mValue
						self.mNetwork.mWeightOutput[i][self.mNo] -= (self.mNetwork.mRate1*change + self.mNetwork.mRate2*self.mNetwork.mLastChangeOutput[i][self.mNo])
						self.mNetwork.mLastChangeOutput[i][self.mNo] = change
					}
				}else{
					for i:=0;i<self.mInputCount;i++{
						change := err * dsigmoid(self.mValue) * self.mNetwork.mHiddenLayer[i].mValue
						self.mNetwork.mWeightOutput[i][self.mNo] -= (self.mNetwork.mRate1*change + self.mNetwork.mRate2*self.mNetwork.mLastChangeOutput[i][self.mNo])
						self.mNetwork.mLastChangeOutput[i][self.mNo] = change
					}
				}

			}
		}
	}()
}


