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
date: 2021-11-02
author: matz
Test DASSH material class
"""
########################################################################
import os
import pytest
import numpy as np
from dassh import Material


def test_material_from_coeff():
    """Try loading material properties by coefficient dictionaries"""
    assert hasattr(Material('d9'), 'thermal_conductivity')
    assert hasattr(Material('ht9'), 'thermal_conductivity')

    # Define a custom dictionary and use it
    c = {'thermal_conductivity': [0.05, 1.0, 2.0],
         'heat_capacity': [480.0, 0.5],
         'density': [1200.0, 0.2, 0.03],
         'viscosity': [1.0, 1.0, 1.0],
         'beta': [0.002]
         }
    m = Material('test_material', coeff_dict=c)
    for prop in ['thermal_conductivity', 'heat_capacity',
                 'density', 'viscosity']:
        assert hasattr(m, prop)
        # Try getting a value from the correlation; should be
        # a float, and should be greater than 0
        assert getattr(m, prop) > 0.0

    # Check the results for one of the values
    m.update(100.0)
    assert m.heat_capacity == pytest.approx(480 + 50.0)
    assert type(m.beta) == float
    assert m.beta == 0.002


def test_material_from_table():
    """Try loading material properties stored exclusively in CSV
    tables in the data/ directory"""
    coolants = ['water', 'sodium', 'potassium', 'nak',
                'lead', 'bismuth', 'lbe']
    temps = [300.0, 500.0, 500.0, 500.0, 800.0, 800.0, 800.0]
    for i in range(len(coolants)):
        mat = Material(coolants[i], temps[i])
        for prop in ['thermal_conductivity', 'heat_capacity',
                     'density', 'viscosity']:
            assert hasattr(mat, prop)
            # Try getting a value from the correlation; should be
            # a float, and should be greater than 0
            assert getattr(mat, prop) > 0.0


def test_error_table_non_positive_val(testdir, caplog):
    """Test error when table has negative value"""
    f = os.path.join(testdir, 'test_inputs', 'custom_mat-3.csv')
    # with pytest.raises(SystemExit):
    #     Material('badbad', from_file=f)
    with pytest.raises(SystemExit):
        Material('badbad', from_file=f)     
    assert 'Non-positive or missing values detected in material data ' in caplog.text


def test_material_coeff_from_file(testdir):
    """Try loading material property correlation coeffs from CSV"""
    filepath = os.path.join(testdir, 'test_inputs', 'custom_mat.csv')
    m = Material('sodium', from_file=filepath)
    m.update(623.15)
    for prop in ['thermal_conductivity', 'heat_capacity',
                 'density', 'viscosity']:
        assert hasattr(m, prop)
        assert getattr(m, prop) > 0.0


def test_failed_material(caplog):
    """Make sure that the Material class fails with bad input"""
    with pytest.raises(SystemExit):
        Material('candycorn')
    assert 'material candycorn' in caplog.text


def test_bad_temperature(caplog):
    """Make sure Material throws error for negative temperatures"""
    m = Material('sodium')
    with pytest.raises(SystemExit):
        m.update(0.0)
    assert 'must be > 0; given' in caplog.text


def test_bad_property(caplog):
    """Make sure Material throws error for negative temperatures"""
    # Define a custom dictionary and use it
    c = {'thermal_conductivity': [0.05, 1.0, 2.0],
         'heat_capacity': [480.0, 0.5],
         'density': [1200.0, 0.2, 0.03],
         'viscosity': [300.0, -1.0],
         }
    m = Material('test', coeff_dict=c)
    with pytest.raises(SystemExit):
        m.update(400.0)
    assert 'viscosity must be > 0; given' in caplog.text


def test_sodium_interpolated_value():
    """Test that DASSH properly interpolates sodium properties"""
    rho_400 = 919.0
    rho_500 = 897.0
    ans = np.average([rho_400, rho_500])
    sodium = Material('sodium', temperature=450.0)
    print('ans =', ans)
    print('res =', sodium.density)
    err = (sodium.density - ans) / ans
    print('err = ', err)
    assert ans == pytest.approx(sodium.density)


def test_table_with_missing_values(testdir, caplog):
    """Test that DASSH interpolates properties
    with missing or zero values"""
    filepath = os.path.join(testdir, 'test_inputs', 'custom_mat-2.csv')
    with pytest.raises(SystemExit):
        Material('missing_sodium', from_file=filepath)    
    assert 'Non-positive or missing values detected in material data ' in caplog.text
    
def __test_property(mat, t_range, correct_values, property):
    """
    Function to test properties
    """
    for i in np.arange(0,len(t_range)-1):
        mat.update(np.average([t_range[i], t_range[i+1]]))
        assert correct_values[i] == pytest.approx(getattr(mat, property))
        
def test_lead():
    """Test that DASSH properly finds the properties for lead"""
    temperature_range = {
        'density': range(650, 2000, 50), # 600.60-2021 K
        'heat_capacity': range(650, 2000, 50),  # 600.60-2000 K
        'viscosity': range(650, 1450, 50),  # 600.60-1475K
        'thermal_conductivity': range(650, 1300, 50)}   # 600.60-1300 K
    
    properties = {
        'density': [10577.337500000001, 10513.3625, 10449.3875, 10385.412499999999, 10321.4375,
                    10257.462500000001, 10193.4875, 10129.5125, 10065.537499999999, 10001.5625,
                    9937.587500000001, 9873.6125, 9809.6375, 9745.662499999999, 9681.6875, 9617.712500000001,
                    9553.7375, 9489.7625, 9425.787499999999, 9361.8125, 9297.837500000001, 9233.8625, 9169.8875,
                    9105.912499999999, 9041.9375, 8977.962500000001, 8913.9875],
        'heat_capacity': [146.6556, 145.72379999999998, 144.78475, 143.85845, 142.95925, 142.0977, 141.28175, 
                          140.5174, 139.80935, 139.1612, 138.5759, 138.0557, 137.60244999999998, 137.21775, 
                          136.90275, 136.6585, 136.48595, 136.38575, 136.35845, 136.40460000000002, 136.5247,
                          136.7191, 136.98805, 137.33190000000002, 137.7509, 138.24525],        
        'viscosity': [0.002225864, 0.0019938655000000002, 0.0018118085, 0.001665729, 0.0015462995, 
                      0.0014470849999999999, 0.00136352, 0.001292287, 0.0012309255, 0.0011775745, 
                      0.001130804, 0.0010894985000000001, 0.0010527765, 0.0010199340000000001,
                      0.0009904002999999999, 0.00096371005],     
        'thermal_conductivity': [16.625, 17.174999999999997, 17.725, 18.275, 18.825000000000003, 
                                 19.375, 19.924999999999997, 20.475, 21.025, 21.575000000000003, 
                                 22.125, 22.674999999999997]}
    
    m = Material('lead', temperature=650)
    for prop_name, correct_values in properties.items():
        __test_property(m, temperature_range[prop_name], correct_values, prop_name)

      
def test_lbe():
    """Test that DASSH properly finds the properties for lbe"""
    temperature_range = {
        'density': range(400, 1900, 50), 
        'heat_capacity': range(400, 1900, 50),  
        'viscosity': range(400, 1300, 50),
        'thermal_conductivity': range(400, 1200, 50)} 
          
    properties = {
        'density': [10515.474999999999, 10450.825, 10386.175, 10321.525000000001, 10256.875, 10192.224999999999, 10127.575, 10062.925, 9998.275000000001, 9933.625, 9868.974999999999, 9804.325, 9739.675, 9675.025000000001, 9610.375, 9545.724999999999, 9481.075, 9416.425, 9351.775000000001, 9287.125, 9222.474999999999, 9157.825, 9093.175, 9028.525000000001, 8963.875, 8899.224999999999, 8834.575, 8769.925, 8705.275000000001, 8640.625],         
        'heat_capacity': [147.7697, 146.8752, 145.9024, 144.89855, 143.89265, 142.9032, 141.9425, 141.01905, 140.1388, 139.30605, 138.524, 137.795, 137.12079999999997, 136.50285, 135.94225, 135.43985, 134.99635, 134.6123, 134.2881, 134.02415, 133.82085, 133.67835, 133.59685000000002, 133.57665, 133.61784999999998, 133.72050000000002, 133.8848, 134.11085, 134.39864999999998, 134.74835000000002],
        'viscosity': [0.002946945, 0.0024358, 0.002089184, 0.0018411185, 0.0016560605000000002, 0.001513399, 0.0014004605, 0.0013090735000000002, 0.0012337585, 0.001170717, 0.001117241, 0.0010713525, 0.0010315749999999999, 0.0009967865000000001, 0.0009661208000000001, 0.0009388979, 0.0009145780999999999],   
        'thermal_conductivity': [9.73847, 10.443245000000001, 11.136495, 11.81822, 12.48842, 13.147095, 13.794245, 14.429870000000001, 15.05397, 15.666545, 16.267595, 16.857120000000002, 17.43512, 18.001595000000002, 18.556545]
    }
    
    m = Material('lbe', temperature=400)
    for prop_name, correct_values in properties.items():
        __test_property(m, temperature_range[prop_name], correct_values, prop_name)
    
    
def test_bismuth():
    """Test that DASSH properly finds the properties for bismuth"""
    temperature_range = {
        'density': range(550, 1800, 50), 
        'heat_capacity': range(550, 1800, 50),  
        'viscosity': range(550, 1300, 50),
        'thermal_conductivity': range(550, 1000, 50)} 
    properties = {
        'density': [10023.5, 9962.5, 9901.5, 9840.5, 9779.5, 9718.5, 9657.5, 9596.5, 9535.5, 9474.5, 9413.5, 9352.5, 9291.5, 9230.5, 9169.5, 9108.5, 9047.5, 8986.5, 8925.5, 8864.5, 8803.5, 8742.5, 8681.5, 8620.5, 8559.5],   
        'heat_capacity': [143.46120000000002, 140.38575, 138.03565, 136.21665000000002, 134.79545000000002, 133.6782, 132.79715, 132.1024, 131.55665, 131.13145, 130.80485, 130.55965, 130.38225, 130.2618, 130.18955, 130.15834999999998, 130.16230000000002, 130.19655, 130.25705, 130.34044999999998, 130.44389999999999, 130.56490000000002, 130.7014, 130.85165, 131.01409999999998],  
        'viscosity': [0.0017375955, 0.0015572415, 0.0014186805, 0.001309308, 0.0012210295, 0.001148436, 0.00108779, 0.001036434, 0.000992432, 0.00095434255, 0.000921072, 0.0008917772499999999, 0.0008657983, 0.0008426116500000001],
        'thermal_conductivity': [12.802499999999998, 13.2775, 13.752500000000001, 14.2275, 14.7025, 15.177499999999998, 15.6525, 16.127499999999998]
    }

    m = Material('bismuth', temperature=550)
    for prop_name, correct_values in properties.items():
        __test_property(m, temperature_range[prop_name], correct_values, prop_name)
        
def test_sodium():
    temperature_range = {
        'density': range(400, 2500, 100), 
        'heat_capacity': range(400, 2500, 100),  
        'viscosity': range(400, 2400, 100),
        'thermal_conductivity': range(400, 2400, 100)
        }
    properties = {
        'density':  [908.0, 885.5, 863.0, 840.0, 816.5, 793.0, 768.5, 744.0, 719.0,
                            693.0, 666.5, 639.5, 611.5, 582.5, 552.5, 520.5, 486.5, 450.0,
                            409.0, 361.0, 287.0, 229.0],
        'heat_capacity': [ 1353.0, 1317.5, 1289.0, 1268.5, 1256.0, 1252.0, 
                                1256.5, 1270.0, 1292.0, 1322.5, 1362.0, 1410.5, 1468.5, 
                                1537.0, 1617.5, 1712.5, 1845.0, 2058.0, 2440.0, 3352.0],
        'viscosity': [0.0005070000000000001, 0.000368, 0.0002925, 0.0002455, 0.000214, 
                            0.000191, 0.0001735, 0.0001595, 0.00014800000000000002, 
                            0.00013900000000000002, 0.0001315, 0.000125, 0.0001195, 0.00011449999999999999,
                            0.00010999999999999999, 0.000106, 0.0001025, 9.95e-05, 9.65e-05, 9.35e-05],
        'thermal_conductivity': [83.655, 76.89500000000001, 70.85, 65.45, 60.620000000000005, 56.290000000000006,
                                52.39, 48.849999999999994, 45.595, 42.555, 39.66, 36.84, 34.025, 31.145,
                                28.125, 24.89, 21.375, 17.509999999999998, 13.225000000000001, 8.445]
        }
    
    m = Material('sodium', temperature=400)
    for prop_name, correct_values in properties.items():
        __test_property(m, temperature_range[prop_name], correct_values, prop_name)

def test_potassium():
    temperature_range = {
        'density': np.arange(373.15, 1473.15, 50), 
        'heat_capacity': np.arange(373.15, 1473.15, 50),  
        'viscosity': np.arange(373.15, 1473.15, 50),
        'thermal_conductivity': np.arange(373.15, 1173.15, 50)
        }
    properties = {
        'density': [813.9, 802.65, 791.25, 779.75, 768.15, 756.45, 744.6500000000001,
                        732.7, 720.6500000000001, 708.5, 696.3, 684.0, 671.5999999999999,
                        659.15, 646.5999999999999, 633.95, 621.2, 608.4, 595.55, 
                        582.6500000000001, 569.7, 556.7],        
        'heat_capacity': [800.5634384, 789.07905095, 779.8911223, 772.9996524999999, 
                                768.4046415, 766.1060893, 766.1039959, 768.3983613, 772.9891855, 
                                779.8764685, 789.06021035, 800.540411, 814.3170704, 830.39018865, 
                                848.7597657, 869.42580155, 892.3882962499999, 917.6472497, 
                                945.20266195, 975.0545330499999, 1007.20286315, 1041.6476519999999],
        'viscosity': [0.0004289, 0.00035800000000000003, 0.00030975, 0.00027499999999999996, 
                            0.0002488, 0.00022675, 0.0002061, 0.0001881, 0.00017355, 0.0001615, 
                            0.00015135, 0.0001427, 0.0001352, 0.00012865, 0.0001229, 0.00011779999999999999, 
                            0.0001132, 0.00010905, 0.0001053, 0.00010185, 9.87e-05, 9.575e-05],
        'thermal_conductivity': [51.0, 48.75, 46.75, 45.099999999999994, 43.4, 41.599999999999994, 
                                40.05, 38.55, 37.099999999999994, 35.7, 34.3, 32.95, 31.65, 30.35, 
                                29.049999999999997, 27.799999999999997]
    }   
    m = Material('potassium', temperature=400)
    for prop_name, correct_values in properties.items():
        __test_property(m, temperature_range[prop_name], correct_values, prop_name)
        
def test_nak():
    temperature_range = {
        'density': np.arange(323.15, 1473.15, 50), 
        'heat_capacity': np.arange(323.15, 1473.15, 50),  
        'viscosity': np.arange(323.15, 1473.15, 50),
        'thermal_conductivity': np.arange(323.15, 1173.15, 50)
        }
    properties = {
        'density': [861.25, 849.9000000000001, 838.45, 826.9000000000001, 815.25, 
                        803.5, 791.6500000000001, 779.7, 767.6500000000001, 755.55,
                        743.35, 731.05, 718.7, 706.3, 693.8, 681.25, 668.65, 655.95, 
                        643.25, 630.5, 617.6500000000001, 604.75, 591.8499999999999],
        'heat_capacity': [884.60141955, 878.94853025, 875.01222815, 872.79251335, 
                                872.2893858499999, 873.5028455500001, 876.4328925, 881.07952675,
                                887.44274825, 895.522557, 905.318953, 916.8319362499999, 930.06150675,
                                945.0076645500001, 961.6704096000001, 980.04974185, 1000.1456614, 
                                1021.958168, 1045.487262, 1070.7329435, 1097.695212, 1126.374068, 1156.769511],
        'viscosity': [0.00061685, 0.00048295, 0.00039945, 0.00034305, 0.00030270000000000004, 0.00027255,
                            0.0002492, 0.0002267, 0.00020545, 0.00018899999999999999, 0.00017545, 0.0001641,
                            0.0001544, 0.00014605, 0.00013885, 0.0001325, 0.00012685000000000002, 0.0001218, 
                            0.00011725, 0.00011315000000000001, 0.0001094, 0.0001059, 0.0001027],
        'thermal_conductivity': [22.799999999999997, 23.6, 24.35, 24.95, 25.4, 25.8, 26.1, 26.25, 26.3,
                                26.200000000000003, 26.0, 25.75, 25.35, 24.85, 24.25, 23.5, 22.65]
    }
    m = Material('nak', temperature=400)
    for prop_name, correct_values in properties.items():
        __test_property(m, temperature_range[prop_name], correct_values, prop_name)
        
def test_ss304():
    temperature_range = {
        'density': np.arange(300, 1700, 100), 
        'heat_capacity': np.arange(300, 1700, 100),  
        'thermal_conductivity':np.arange(300, 1700, 100),
        }
    properties = {
        'density': [7877.0, 7841.5, 7803.0, 7762.5, 7720.0, 7675.0, 7627.5, 
                        7577.5, 7525.5, 7471.5, 7415.0, 7356.0, 7295.0, 7231.5],
        
        'heat_capacity': [516.724, 530.1128, 543.5016, 557.0996, 570.6976, 584.0864, 
                                597.4752, 610.864, 624.462, 638.06, 651.4488, 664.8376000000001,
                                678.4356, 692.0336],
        
        'thermal_conductivity': [13.780000000000001, 15.395, 17.009999999999998, 18.630000000000003, 
                                20.25, 21.865000000000002, 23.48, 25.1, 26.72, 28.335, 29.950000000000003,
                                31.57, 33.19, 34.805]
    }
    m = Material('ss304', temperature=300)
    for prop_name, correct_values in properties.items():
        __test_property(m, temperature_range[prop_name], correct_values, prop_name)
        

                
def test_ss316():
    temperature_range = {
        'density': np.arange(300, 1700, 100),
        'heat_capacity': np.arange(300, 1700, 100),
        'thermal_conductivity': np.arange(300, 1700, 100)
    }
    
    properties = {
        'density' : [7932.0, 7887.0, 7841.0, 7794.5, 7747.0, 7698.5, 7649.0,
                        7599.0, 7548.5, 7497.0, 7445.0, 7392.0, 7338.0, 7283.5], 
        'heat_capacity': [505.42719999999997, 518.816, 531.9956, 545.1752, 
                                558.5640000000001, 571.9528, 585.1324, 598.312,
                                611.7008000000001, 625.0896, 638.2692, 651.4488, 
                                664.8376000000001, 678.2264],     
        'thermal_conductivity': [14.745000000000001, 16.315, 17.89, 19.465, 21.035,
                                22.605, 24.175, 25.745, 27.315, 28.885, 30.46, 
                                32.035, 33.605000000000004, 35.175]
    }
    
    m = Material('ss316', temperature=300)
    for prop_name, correct_values in properties.items():
        __test_property(m, temperature_range[prop_name], correct_values, prop_name)
    
    
def test_interpolation_with_missing_value(testdir):
    """
    Tests that interpolation is correctly performed
    in case of missing value in a user-defined table
    """
    f = os.path.join(testdir, 'test_inputs', 'custom_missing.csv')
    mat = Material('test_mat', from_file=f)  
    mat.update(1700)
    assert mat.density == pytest.approx(597.0)   
    
    f = os.path.join(testdir, 'test_inputs', 'custom_missing-2.csv')
    mat = Material('test_mat', from_file=f)
    mat.update(1700)
    assert mat.density == pytest.approx(595.0)
