import utils.dlc_bci as bci

train_input , train_target = bci.load(root='../../data_bci', one_khz = False)
print ( str ( type ( train_input ) ) , train_input.size())
print ( str ( type ( train_target ) ) , train_target.size())
test_input , test_target = bci.load ( root = '../../data_bci', train = False, one_khz = False)
print ( str ( type ( test_input ) ) , test_input.size())
print ( str ( type ( test_target ) ) , test_target.size())