python cocogan_train.py --config ../exps/unitbank/hair.yaml --log ../logs --gpu 0

python cocogan_train.py --config ../exps/unitbank/hair_101000.yaml --log ../logs --gpu 6
python cocogan_train.py --config ../exps/unitbank/hair_101001.yaml --log ../logs --gpu 4
python cocogan_train.py --config ../exps/unitbank/hair_101002.yaml --log ../logs --gpu 3
python cocogan_train.py --config ../exps/unitbank/hair_101003.yaml --log ../logs --gpu 1

python cocogan_train.py --config ../exps/unitbank/hair_101100.yaml --log ../logs --gpu 3

---- blond, brown, black ----
fix loss ratio, adjust structure
101101:  3,3,1,1,3,3  4,2 ok
101102:  3,3,1,1,3,3  6,0
101103:  3,3,1,1,3,3  2,4

python cocogan_train.py --config ../exps/unitbank/hair_101101.yaml --log ../logs --gpu 0
#python cocogan_train.py --config ../exps/unitbank/hair_101102.yaml --log ../logs --gpu 1
#python cocogan_train.py --config ../exps/unitbank/hair_101103.yaml --log ../logs --gpu 2

---- blond, brown, black, Gray ----
fix loss ratio, adjust structure
101104:  3,3,1,1,3,3  4,2 
python cocogan_train.py --config ../exps/unitbank/hair_101104.yaml --log ../logs --gpu 2


---- cat: 3 to try ---
101100 replace ResGen2 -> ResGen, add instance norm
python cocogan_train.py --config ../exps/unitbank/cat.yaml --log ../logs --gpu 4
python cocogan_train.py --config ../exps/unitbank/cat_101100.yaml --log ../logs --gpu 5

---- dog: 3 to try ---
python cocogan_train.py --config ../exps/unitbank/dog.yaml --log ../logs --gpu 2


---- sharing all front ----
python cocogan_train.py --config ../exps/unitbank/hair_101200.yaml --log ../logs --gpu 6
python cocogan_train.py --config ../exps/unitbank/hair_101201.yaml --log ../logs --gpu 7
