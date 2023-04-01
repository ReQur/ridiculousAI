package main

import "fmt"

func main() {
	var net = define([][]float64{{0, 1}, {0, 1}}, 1, hardlim)

	var inputData = [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	var target = [][]float64{{0}, {1}, {1}, {1}}

	net = train(net, inputData, target)

	fmt.Println(sim(net, inputData))
}
