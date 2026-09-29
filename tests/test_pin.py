########################################################################
# Copyright 2021, UChicago Argonne, LLC
#
# Licensed under the BSD-3 License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a
# copy of the License at
#
#     https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
########################################################################
"""
date: 2020-02-06
author: matz
Test the pin.py module and the PinLattice object
"""
########################################################################
import numpy as np
import pytest
import dassh


def test_count_pins():
    """Test the count_pins method from the pin module."""
    assert dassh.count_pins(5) == 61
    assert dassh.count_pins(1) == 1
    assert dassh.count_pins(10) == 271


def test_get_start_pin():
    """Test the get_start_pin method from the pin module."""
    assert dassh.get_start_pin(1) == 1
    assert dassh.get_start_pin(2) == 2
    assert dassh.get_start_pin(10) == 218


def test_get_end_pin():
    """Test the get_end_pin method from the pin module."""
    assert dassh.get_end_pin(1) == 1
    assert dassh.get_end_pin(2) == 7
    assert dassh.get_end_pin(10) == 271


def test_get_corners():
    """Test the get_corners method from the pin module."""
    with pytest.raises(ZeroDivisionError):
        dassh.get_corners(1)
    assert len(dassh.get_corners(5)) == 6
    assert len(dassh.get_corners(10)) == 6
    assert np.array_equal(dassh.get_corners(2),
                          np.array([2, 3, 4, 5, 6, 7]))


def test_map(pinlattice_2ring, pinlattice_5ring):
    """Test that the pin map matches expected result."""
    test2 = pinlattice_2ring.make_pin_map(2)
    test5 = pinlattice_5ring.make_pin_map(5)
    assert np.array_equal(test2, np.array([[2, 3, 0],
                                           [7, 1, 4],
                                           [0, 6, 5]]))
    assert np.array_equal(test5,
                          np.array([[38, 39, 40, 41, 42, 0, 0, 0, 0],
                                    [61, 20, 21, 22, 23, 43, 0, 0, 0],
                                    [60, 37, 8, 9, 10, 24, 44, 0, 0],
                                    [59, 36, 19, 2, 3, 11, 25, 45, 0],
                                    [58, 35, 18, 7, 1, 4, 12, 26, 46],
                                    [0, 57, 34, 17, 6, 5, 13, 27, 47],
                                    [0, 0, 56, 33, 16, 15, 14, 28, 48],
                                    [0, 0, 0, 55, 32, 31, 30, 29, 49],
                                    [0, 0, 0, 0, 54, 53, 52, 51, 50]]))


def test_neighbors(pinlattice_2ring_map, pinlattice_5ring_map):
    """Test that the pin lattice neighbors match expected result."""
    # Do indirect tests on large array, direct comparison on small array.
    test2 = pinlattice_2ring_map.map_pin_neighbors()
    test5 = pinlattice_5ring_map.map_pin_neighbors()
    ring5 = 5
    # Array rows and cols
    assert test5.shape == (pinlattice_5ring_map.n_pin, 6)
    # Interior pins should touch 6 others (will show up 6x in array)
    assert all([len(test5[test5 == i]) == 6 for i in
                range(1, dassh.get_end_pin(ring5 - 1))])
    # All corners on outer ring should touch only 3 pins
    assert all([len(test5[test5 == i]) == 3 for i in
                dassh.get_corners(ring5)])
    # Should be 6 x (n_ring-2) side pins, each touching 4 other pins
    # Therefore, each will have two zeros in their neighbors row
    i = 0
    for row in test5:
        if len(row[row == 0]) == 2:
            i += 1
    assert i == 6 * (ring5 - 2)
    # Check direct comparison on small array
    assert np.array_equal(test2, np.array([[2, 3, 4, 5, 6, 7],
                                           [0, 0, 3, 1, 7, 0],
                                           [0, 0, 0, 4, 1, 2],
                                           [3, 0, 0, 0, 5, 1],
                                           [1, 4, 0, 0, 0, 6],
                                           [7, 1, 5, 0, 0, 0],
                                           [0, 2, 1, 6, 0, 0]]))


def test_pin_xy(pinlattice_2ring_neighbors):  # , pinlattice_5ring):
    """Test the pin X-Y coordinates."""
    pitch = 1.0  # from conftest
    test2 = pinlattice_2ring_neighbors.map_pin_xy(2, pitch, (0.0, 0.0))
    ans = np.array([[0.0, 0.0],
                    [0.0, 1.0],
                    [np.cos(np.pi / 6), np.sin(np.pi / 6)],
                    [np.cos(-np.pi / 6), np.sin(-np.pi / 6)],
                    [0.0, -1.0],
                    [np.cos(7 * np.pi / 6), np.sin(7 * np.pi / 6)],
                    [np.cos(5 * np.pi / 6), np.sin(5 * np.pi / 6)]])
    assert np.allclose(test2, ans)


def test_map_with_hole(pinlattice_5ring_2removed: dassh.PinLattice):
    """
    Test that the pin map matches expected result when a central hole is
    considered.

    Parameters
    ----------
    pinlattice_5ring_2removed : dassh.PinLattice
        Pin lattice data considering a central hole replacing the first 2 rings
        of pins, provided by the fixture pinlattice_5ring_2removed.
    """
    assert np.array_equal(pinlattice_5ring_2removed.map,
                          np.array([[31, 32, 33, 34, 35,  0,  0,  0,  0],
                                    [54, 13, 14, 15, 16, 36,  0,  0,  0],
                                    [53, 30,  1,  2,  3, 17, 37,  0,  0],
                                    [52, 29, 12,  0,  0,  4, 18, 38,  0],
                                    [51, 28, 11,  0,  0,  0,  5, 19, 39],
                                    [ 0, 50, 27, 10,  0,  0,  6, 20, 40],
                                    [ 0,  0, 49, 26,  9,  8,  7, 21, 41],
                                    [ 0,  0,  0, 48, 25, 24, 23, 22, 42],
                                    [ 0,  0,  0,  0, 47, 46, 45, 44, 43],]))


def test_neighbors_hole(pinlattice_5ring_2removed: dassh.PinLattice):
    """
    Test that the pin lattice neighbors match expected result when a central
    hole is considered.

    Parameters
    ----------
    pinlattice_5ring_2removed : dassh.PinLattice
        Pin lattice data considering a central hole replacing the first 2 rings
        of pins, provided by the fixture pinlattice_5ring_2removed.
    """
    adj = pinlattice_5ring_2removed.adj
    ring = 5
    n_ring_rem = 2
    n_pins_rem = dassh.count_pins(n_ring_rem)
    # Array rows and cols
    assert adj.shape == (pinlattice_5ring_2removed.n_pin, 6)
    # Check on inner-corner inner-edge nearest pins. Pins lying on the
    # inner hexagon hole's vertices have 5 nearby pins.
    # get_end_pin function calculate the number of pins without considering the
    # number of pins removed by the hole. Need to shift indexes according to
    # n_pins_rem
    assert all([len(adj[adj == i]) == 5 for i in
                    range(1, dassh.get_end_pin(n_ring_rem + 1)-n_pins_rem, 2)])
    # The pins on the inner hexagonal side of the hole have 4 nearby pins.
    assert all([len(adj[adj == i]) == 4 for i in
                    range(2, dassh.get_end_pin(n_ring_rem + 1)-n_pins_rem, 2)])
    # Interior pins should touch 6 others (will show up 6x in array)
    assert all([len(adj[adj == i]) == 6 for i in
                range(dassh.get_end_pin(n_ring_rem + 1) - n_pins_rem + 1,
                      dassh.get_end_pin(ring - 1) - n_pins_rem)])
    # All corners on outer ring should touch only 3 pins
    assert all([len(adj[adj == i]) == n_ring_rem + 1 for i in
                dassh.get_corners(ring) - n_pins_rem])
    # Direct comparison on a portion of the array
    assert np.array_equal(
        adj[:dassh.get_end_pin(ring - 1) - n_pins_rem, :],
        np.array([
            [13, 14,  2,  0, 12, 30],
            [14, 15,  3,  0,  0,  1],
            [15, 16, 17,  4,  0,  2],
            [ 3, 17, 18,  5,  0,  0],
            [ 4, 18, 19, 20,  6,  0],
            [ 0,  5, 20, 21,  7,  0],
            [ 0,  6, 21, 22, 23,  8],
            [ 0,  0,  7, 23, 24,  9],
            [10,  0,  8, 24, 25, 26],
            [11,  0,  0,  9, 26, 27],
            [29, 12,  0, 10, 27, 28],
            [30,  1,  0,  0, 11, 29],
            [31, 32, 14,  1, 30, 54],
            [32, 33, 15,  2,  1, 13],
            [33, 34, 16,  3,  2, 14],
            [34, 35, 36, 17,  3, 15],
            [16, 36, 37, 18,  4,  3],
            [17, 37, 38, 19,  5,  4],
            [18, 38, 39, 40, 20,  5],
            [ 5, 19, 40, 41, 21,  6],
            [ 6, 20, 41, 42, 22,  7],
            [ 7, 21, 42, 43, 44, 23],
            [ 8,  7, 22, 44, 45, 24],
            [ 9,  8, 23, 45, 46, 25],
            [26,  9, 24, 46, 47, 48],
            [27, 10,  9, 25, 48, 49],
            [28, 11, 10, 26, 49, 50],
            [52, 29, 11, 27, 50, 51],
            [53, 30, 12, 11, 28, 52],
            [54, 13,  1, 12, 29, 53]]))
