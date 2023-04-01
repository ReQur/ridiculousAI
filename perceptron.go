package main

type perceptron struct {
	wights  []float64
	f       func(float64) float64
	padding float64
}

type trainParameters struct {
	epochs int
	r      float64
}

type network struct {
	neurons          []perceptron
	_trainParameters trainParameters
}

func define(
	PR [][]float64,
	S int,
	f func(float64) float64,
) network {
	var _network network
	var _weights []float64
	for i := 0; i < len(PR); i++ {
		_weights = append(_weights, 0.0)
	}
	for i := 0; i < S; i++ {
		var w = make([]float64, len(_weights))
		copy(w, _weights)
		_network.neurons = append(_network.neurons, perceptron{wights: w, padding: -1, f: f})
	}
	_network._trainParameters.epochs = 100
	_network._trainParameters.r = 1
	return _network
}

func neuronActivation(
	p perceptron,
	input []float64,
) float64 {
	var answer float64
	for i := 0; i < len(input); i++ {
		answer += p.wights[i] * input[i]
	}
	answer += p.padding

	return p.f(answer)
}

func sim(
	net network,
	P [][]float64,
) [][]float64 {
	var answer [][]float64
	for i := 0; i < len(P); i++ {
		var _answer []float64
		for j := 0; j < len(net.neurons); j++ {
			_answer = append(_answer, neuronActivation(net.neurons[j], P[i]))
		}
		answer = append(answer, _answer)
	}
	return answer
}

func train(
	net network,
	P, T [][]float64,
) network {
	var errorFlag bool = false
	for e := 0; e < net._trainParameters.epochs; e++ {
		for n := 0; n < len(net.neurons); n++ {
			for i := 0; i < len(P); i++ {
				var _ans = neuronActivation(net.neurons[n], P[i])
				var err = T[i][n] - _ans
				if err == 0 {
					continue
				}
				errorFlag = true
				var _weights []float64
				for w := 0; w < len(net.neurons[n].wights); w++ {
					var _weight = net.neurons[n].wights[w] + net._trainParameters.r*err*P[i][w]
					_weights = append(_weights, _weight)
				}
				net.neurons[n].wights = _weights
			}
		}
		if !errorFlag {
			break
		}
		errorFlag = false
	}
	return net
}
