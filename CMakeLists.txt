# cmake_minimum_required(VERSION <specify CMake version here>)
project(word2vec_master C)

set(CMAKE_C_STANDARD 11)

add_executable(word2vec_master
        compute-accuracy.c
        distance.c
        word-analogy.c
        word2phrase.c
        word2vec.c)
