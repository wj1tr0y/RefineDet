cur_dir="$HOME/pedestrian/RefineDet/data/zhili_coco_people"
root_dir=$cur_dir/../..

cd $root_dir

redo=false
data_root_dir="$HOME/data/zhili_coco_people"
dataset_name="zhili_coco_people"
mapfile="$root_dir/data/$dataset_name/labelmap_coco.prototxt"
anno_type="detection"
label_type="json"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if $redo
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in train 
do
  python $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-type=$label_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $root_dir/data/$dataset_name/$subset.txt $data_root_dir/$db/$dataset_name"_"$subset"_"$db examples/$dataset_name 2>&1 | tee $root_dir/data/$dataset_name/$subset.log
done
