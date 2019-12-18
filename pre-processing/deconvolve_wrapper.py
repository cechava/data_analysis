import os

sess_list = ['20191115','20191116','20191121','20191124','20191125','20191126','20191127']
#,'20191007']

for sess in sess_list:
	print(sess)
	try:
		os.system('python run_spike_dcnv.py -i JC120 -S %s -A FOV1_zoom4p0x -R all_combined -Y suite2p_analysis102\
		 -T traces102'%(sess))
	except:
		break
