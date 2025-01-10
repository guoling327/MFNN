#python training.py --filter_type band --net BernNet
#python training.py --filter_type band --net OurSineEncoding0

python training.py --filter_type comb --net MDSGNN
python training.py --filter_type low --net MDSGNN
python training.py --filter_type high --net MDSGNN
python training.py --filter_type rejection --net MDSGNN
python training.py --filter_type band --net MDSGNN



### other models demo
#for model in ChebNet GcnNet GatNet ARMANet GPRNet BernNet
#do
#    python training.py --filter_type band --net $model
#done
##
