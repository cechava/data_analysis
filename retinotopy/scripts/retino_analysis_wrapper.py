import os

sess_list = ['20191016','20191019','20191021','20191023','20191025','20191029','20191101','20191106']

for sess in sess_list:
	print(sess)
	result1 = os.system('python run_retino_analysis.py -i JC113 -S %s -A FOV1_zoom4p0x -T traces102 -R retino_run1'\
		%(sess))
	assert result1 == 0
	result2 =os.system('python run_retino_analysis.py -i JC113 -S %s -A FOV1_zoom4p0x -T traces102 -R retino_run2'\
		%(sess))
	assert result2 == 0


