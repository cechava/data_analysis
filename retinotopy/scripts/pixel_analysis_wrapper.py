import os

sess_list = ['20191115']
#sess_list = ['20190930','20191003']
#,'20191007']

for sess in sess_list:
	result1 = os.system('python run_pixel_retino_analysis.py -i JC120 -S %s -A FOV1_zoom4p0x -T traces102 \
		-r retino_run1,retino_run2'%sess)

	assert result1 == 0