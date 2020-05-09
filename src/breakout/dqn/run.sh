#!/bin/sh

docker run --rm -it -v ~/workspace/rl-system-trade/src/breakout/dqn/result/:/workspace/result/ dqn/breakout:latest