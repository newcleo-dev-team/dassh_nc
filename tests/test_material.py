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


def test_error_table_negative_val(testdir, caplog):
    """Test error when table has negative value"""
    f = os.path.join(testdir, 'test_inputs', 'custom_mat-3.csv')
    # with pytest.raises(SystemExit):
    #     Material('badbad', from_file=f)
    Material('badbad', from_file=f)
    assert 'Negative values detected in material data ' in caplog.text


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


def test_table_with_missing_values(testdir):
    """Test that DASSH interpolates properties
    with missing or zero values"""
    filepath = os.path.join(testdir, 'test_inputs', 'custom_mat-2.csv')
    m = Material('sodium', from_file=filepath)
    # missing values in heat capacity
    m.update(950.0)
    assert m.heat_capacity == pytest.approx(1252.0)
    # zero values in density; missing values in viscosity
    # linear interp should return average
    m.update(850.0)
    assert m.density == pytest.approx(np.average([828, 805]))
    assert m.viscosity == pytest.approx(np.average([0.000227, 0.000201]))
def _test_property(mat, t_range, correct_values, property):
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
        'density': [10577.3375, 10513.3625, 10449.3875, 10385.4125, 10321.4375,
                       10257.4625, 10193.4875, 10129.5125, 10065.5375, 10001.5625, 
                       9937.5875, 9873.6125, 9809.6375, 9745.6625, 9681.6875, 9617.7125, 
                       9553.7375, 9489.7625, 9425.7875, 9361.8125, 9297.8375, 9233.8625, 
                       9169.8875, 9105.9125, 9041.9375, 8977.9625],
        'heat_capacity': [146.6597440329218, 145.72449453032104, 144.78304308012486, 
                                143.85498154269973, 142.9544693877551, 142.0919458728999, 
                                141.27524418145956, 140.51033486020225, 139.80183309897242, 
                                139.15335185185185, 138.56775210502488, 138.0473231570179, 
                                137.59391441753172, 137.2090330722677, 136.89391735537188, 
                                136.6495921514312, 136.47691163458774, 136.3765922870196, 
                                136.3492386999244, 136.3953639053254, 136.51540552461572, 
                                136.70973868935096, 136.97868645110097, 137.3225282229311, 
                                137.74150666666665, 138.2358333445775],        
        'viscosity': [0.0022172015633682355, 0.0019877928420200973, 0.0018074137908630303,
                            0.0016624621323030578, 0.0015438146911014726, 0.0014451567824505652,
                            0.0013619975329459168, 0.0012910669496039276, 0.0012299342892951547,
                            0.0011767593680168567, 0.0011301262747093605, 0.0010889296063123061,
                            0.0010522950202684787, 0.0010195227165141596, 0.0009900465521585542],     
        'thermal_conductivity': [16.625, 17.174999999999997, 17.725, 18.275, 18.825, 19.375, 
                                19.924999999999997, 20.474999999999998, 21.025, 21.575, 22.125,
                                22.674999999999997]}
    
    m = Material('lead', temperature=650)
    for prop_name, correct_values in properties.items():
        _test_property(m, temperature_range[prop_name], correct_values, prop_name)

      
def test_lbe():
    """Test that DASSH properly finds the properties for lbe"""
    temperature_range = {
        'density': range(400, 1900, 50), 
        'heat_capacity': range(400, 1900, 50),  
        'viscosity': range(400, 1300, 50),
        'thermal_conductivity': range(400, 1200, 50)} 
          
    properties = {
        'density': [10515.475, 10450.825, 10386.175, 10321.525, 10256.875, 10192.225,
                        10127.575, 10062.925, 9998.275, 9933.625, 9868.975, 9804.325, 
                        9739.675, 9675.025, 9610.375, 9545.725, 9481.075, 9416.425, 9351.775, 
                        9287.125, 9222.475, 9157.825, 9093.175, 9028.525, 8963.875, 8899.225, 
                        8834.575, 8769.925, 8705.275],         
        'heat_capacity': [147.78824502595157, 146.88425986842105, 145.90589073129252, 144.89860645085068,
                                143.8904525, 142.89948945473253, 141.93777385552914, 141.01360334287202,
                                140.13284004820937, 139.29972066326533, 138.51736874543465, 137.78812808185407,
                                137.11378513533614, 136.49572082882642, 135.93501620370373, 135.43252730307833,
                                134.9889391139109, 134.60480500288352, 134.28057593894627, 134.0166224173554, 
                                133.81325109649126, 133.67071755601842, 133.58923617643106, 133.56898785903252, 
                                133.61012610946747, 133.71278186957008, 133.87706738342786, 134.10307931214047, 
                                134.3909012596172],
        'viscosity': [0.002912853483877453, 0.0024165941807451205, 0.002077491571265685,
                            0.0018335582888928, 0.0016509324313478202, 0.0015097822322762658, 
                            0.001397825995379052, 0.0013071020305303622, 0.001232248863450807, 
                            0.0011695380123993957, 0.001116304282593233, 0.0010705966775742108, 
                            0.0010309570599959278, 0.000996275563282574, 0.0009656936276816985,
                            0.0009385374423957527, 0.0009142712994326446],   
        'thermal_conductivity': [9.739909375, 10.444684375, 11.137934375, 11.819659375, 12.489859374999998, 
                                13.148534375, 13.795684375, 14.431309375000001, 15.055409375, 15.667984375,
                                16.269034375, 16.858559375000002, 17.436559375, 18.003034375, 18.557984375]
    }
    
    m = Material('lbe', temperature=400)
    for prop_name, correct_values in properties.items():
        _test_property(m, temperature_range[prop_name], correct_values, prop_name)
    
    
def test_bismuth():
    """Test that DASSH properly finds the properties for bismuth"""
    temperature_range = {
        'density': range(550, 1800, 50), 
        'heat_capacity': range(550, 1800, 50),  
        'viscosity': range(550, 1300, 50),
        'thermal_conductivity': range(550, 1000, 50)} 
    properties = {
        'density': [10023.5, 9962.5, 9901.5, 9840.5, 9779.5, 9718.5, 9657.5,
                        9596.5, 9535.5, 9474.5, 9413.5, 9352.5, 9291.5, 9230.5,
                        9169.5, 9108.5, 9047.5, 8986.5, 8925.5, 8864.5, 8803.5,
                        8742.5, 8681.5, 8620.5],   
        'heat_capacity': [143.33756984877127, 140.29722999999998, 137.97060775034294,
                                136.1677861474435, 134.758059157128, 133.64908535353536,
                                132.7741275510204, 132.08398287070855, 131.54173152531231,
                                131.11923280785248, 130.79473415359655, 130.55120679012344,
                                130.3751661611589, 130.25582221990837, 130.18445822760478,
                                130.15397043431827, 130.15852272727273, 130.1932845644814,
                                130.25423000574548, 130.33798208814835, 130.44169121945075,
                                130.5629393491124, 130.69966385609268, 130.85009664986347],  
        'viscosity': [0.0017301154644884355, 0.0015521893367222575, 0.0014151308002842873,
                            0.0013067308431665508, 0.0012191062673424027, 0.0011469668716065987,
                            0.0010866450143416347, 0.0010355261477620402, 0.0009917010377362436,
                            0.0009537459139248721, 0.0009205792430780245, 0.0008913659147939138,
                            0.0008654516209674798, 0.0008423169617084873],
        'thermal_conductivity': [12.802499999999998, 13.2775, 13.7525, 14.2275, 14.7025, 
                                15.177499999999998, 15.6525, 16.127499999999998]
    }

    m = Material('bismuth', temperature=550)
    for prop_name, correct_values in properties.items():
        _test_property(m, temperature_range[prop_name], correct_values, prop_name)
        
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
        _test_property(m, temperature_range[prop_name], correct_values, prop_name)

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
        _test_property(m, temperature_range[prop_name], correct_values, prop_name)
        
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
        _test_property(m, temperature_range[prop_name], correct_values, prop_name)
        
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
        _test_property(m, temperature_range[prop_name], correct_values, prop_name)
        

                
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
        _test_property(m, temperature_range[prop_name], correct_values, prop_name)
    
    
