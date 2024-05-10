#!/usr/bin/env python

import sys
from predict import predict_outcomes


def who_will_win():
    # get arguments from the command line
    if len(sys.argv) != 3:
        print("Usage: python pokemon_model.py <pokemon1_id> <pokemon2_id>")
        sys.exit(1)

    pokemon1_id = int(sys.argv[1])
    pokemon2_id = int(sys.argv[2])

    # predict the outcome
    predict_outcomes(pokemon1_id, pokemon2_id)


if __name__ == "__main__":
    who_will_win()
