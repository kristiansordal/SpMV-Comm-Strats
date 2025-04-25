for i in {1..10}; do
sh $1_test.sh 1a $1
sh $1_test.sh 1b $1
sh $1_test.sh 1c $1
sh $1_test.sh 1d $1
done
