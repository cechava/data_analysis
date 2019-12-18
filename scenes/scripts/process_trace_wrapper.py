import os
import subprocess
#sess_list = ['20190930','20191003']
#,'20191007']
sess_list = ['20191115']

for sess in sess_list:
	print(sess)

	# os.system('python parse_suite2p_trace.py -i JC110 -S %s -A FOV1_zoom4p0x -R all_combined -Y suite2p_analysis102 \
	#  -r retino_run1,retino_run2,scenes_run1,scenes_run2,scenes_run3,scenes_run4,scenes_run5,scenes_run6,scenes_run7,scenes_run8\
	#   -T traces102'%(sess))

	# os.system('python process_trace.py -i JC110 -S %s -A FOV1_zoom4p0x -R all_combined -Y suite2p_analysis102\
	#  -r scenes_run1,scenes_run2,scenes_run3,scenes_run4,scenes_run5,scenes_run6,scenes_run7,scenes_run8 \
	#  -T traces102 -C scenes_combined -m 5'%(sess))

	result1 = os.system('python parse_suite2p_trace.py -i JC120 -S %s -A FOV1_zoom4p0x -R all_combined -Y suite2p_analysis102 \
	 -r retino_run1,retino_run2,scenes_run1,scenes_run2,scenes_run3,scenes_run4,scenes_run5,scenes_run6,scenes_run7,scenes_run8,scenes_run9,scenes_run10\
	  -T traces102'%(sess))

	assert result1 == 0

	result2 = os.system('python process_trace.py -i JC120 -S %s -A FOV1_zoom4p0x -R all_combined -Y suite2p_analysis102\
	 -r scenes_run1,scenes_run2,scenes_run3,scenes_run4,scenes_run5,scenes_run6,scenes_run7,scenes_run8,scenes_run9,scenes_run10 \
	 -T traces102 -C scenes_combined -m 5'%(sess))

	assert result2 == 0