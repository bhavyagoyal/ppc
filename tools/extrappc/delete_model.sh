for (( i = 1; i < 10; i++ ))
do
	find ../../work_dir_py/kitti/pvrcnn/ -iname epoch_$i.pth | xargs rm
done

for (( i = 11; i < 40; i++ ))
do
	find ../../work_dir_py/kitti/pvrcnn/ -iname epoch_$i.pth | xargs rm
done

for (( i = 1; i < 12; i++ ))
do
	find ../../work_dir_py/sbr/ -iname epoch_$i.pth | xargs rm
done

for (( i = 13; i < 36; i++ ))
do
	find ../../work_dir_py/sbr/ -iname epoch_$i.pth | xargs rm
done


