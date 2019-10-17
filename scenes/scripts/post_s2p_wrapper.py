import os

animalid = "JC091"
session = "20190625"
acquisition = "FOV1_zoom4p0x"
all_runs = "'retino_run1,scenes_run1,scenes_run2,scenes_run3,scenes_run4,scenes_run5,scenes_run6'"
# ##JC080_20190619
# good_scene_runs = "'scenes_run1,scenes_run2,scenes_run4,scenes_run5,scenes_run6'"

# ##JC091_20190621
# good_scene_runs = "'scenes_run1,scenes_run3,scenes_run4,scenes_run5,scenes_run6'"

# ##JC097_20190621
# good_scene_runs = "'scenes_run1,'scenes_run2',scenes_run3,scenes_run4,scenes_run5,scenes_run6'"

# ##JC085_20190624
# good_scene_runs = "'scenes_run1,'scenes_run2',scenes_run3,scenes_run5,scenes_run6'"

##JC091_20190625
good_scene_runs = "'scenes_run1,'scenes_run2',scenes_run3,scenes_run4,scenes_run5'"

# ##JC097_20190625
# good_scene_runs = "'scenes_run1,'scenes_run2',scenes_run3,scenes_run4,scenes_run5'"

# ##JC097_20190628
# good_scene_runs = "'scenes_run1,'scenes_run2',scenes_run3,scenes_run4,scenes_run5,scenes_run6'"

# ##JC091_20190701
# good_scene_runs = "'scenes_run1,'scenes_run2',scenes_run3,scenes_run4,scenes_run5,scenes_run6'"

# ##JC097_20190702
# good_scene_runs = "'scenes_run1,'scenes_run2',scenes_run3,scenes_run4,scenes_run5,scenes_run6'"

# ##JC091_20190703
# good_scene_runs = "'scenes_run1,'scenes_run2',scenes_run3,scenes_run4,scenes_run5,scenes_run6'"

# ##JC097_20190704
# good_scene_runs = "'scenes_run1,'scenes_run2',scenes_run3,scenes_run4,scenes_run5,scenes_run6'"

# # ##JC091_20190705
# good_scene_runs = "'scenes_run1,'scenes_run2',scenes_run3,scenes_run4,scenes_run5,scenes_run6'"


# ##JC097_20190705
# good_scene_runs = "'scenes_run1,'scenes_run2',scenes_run3,scenes_run4,scenes_run5,scenes_run6'"

# ##JC097_20190708
# good_scene_runs = "'scenes_run1,'scenes_run2',scenes_run4,scenes_run5,scenes_run6'"

# ##JC085_20190712
# good_scene_runs = "'scenes_run2,scenes_run3,scenes_run4,scenes_run5,scenes_run6'"



# #parse trace
# cmd = "python parse_suite2p_trace.py -i %s -S %s -A %s -r %s -Y suite2p_analysis101 -T traces101_s2p -R all_combined"%(animalid,session,acquisition,all_runs)

# print('**COMMAND**')
# print(cmd)
# print('***********')

# os.system(cmd)


# cmd = "python parse_suite2p_trace.py -i %s -S %s -A %s -r %s -Y suite2p_analysis102 -T traces102_s2p -R all_combined"%(animalid,session,acquisition,all_runs)

# print('**COMMAND**')
# print(cmd)
# print('***********')

# os.system(cmd)


#process trace
cmd = "python process_trace.py -i %s -S %s -A %s -r %s -Y suite2p_analysis101 -T traces101 -C scenes_combined"%(animalid,session,acquisition,good_scene_runs)

print('**COMMAND**')
print(cmd)
print('***********')

os.system(cmd)


cmd = "python process_trace.py -i %s -S %s -A %s -r %s -Y suite2p_analysis102 -T traces102 -C scenes_combined"%(animalid,session,acquisition,good_scene_runs)

print('**COMMAND**')
print(cmd)
print('***********')

os.system(cmd)


# #filter traces

cmd = "python filter_and_average_traces.py -i %s -S %s -A %s -T traces101 -C scenes_combined -f zscore -t 1"%(animalid,session,acquisition)

print('**COMMAND**')
print(cmd)
print('***********')


os.system(cmd)


cmd = "python filter_and_average_traces.py -i %s -S %s -A %s  -T traces102 -C scenes_combined -f zscore -t 1"%(animalid,session,acquisition)

print('**COMMAND**')
print(cmd)
print('***********')


os.system(cmd)


# # #filter responses

# cmd = 'python filter_and_average_responses.py -i %s -S %s -A %s -T traces101 -C scenes_combined -f zscore -t 1 -d df_f'%(animalid,session,acquisition)

# print('**COMMAND**')
# print(cmd)
# print('***********')


# os.system(cmd)

# cmd = 'python filter_and_average_responses.py -i %s -S %s -A %s -T traces102 -C scenes_combined -f zscore -t 1 -d df_f'%(animalid,session,acquisition)

# print('**COMMAND**')
# print(cmd)
# print('***********')


# os.system(cmd)

# cmd = 'python filter_and_average_responses.py -i %s -S %s -A %s -T traces101 -C scenes_combined -f zscore -t 1 -d norm_df'%(animalid,session,acquisition)

# print('**COMMAND**')
# print(cmd)
# print('***********')


# os.system(cmd)

# cmd = 'python filter_and_average_responses.py -i %s -S %s -A %s -T traces102 -C scenes_combined -f zscore -t 1 -d norm_df'%(animalid,session,acquisition)

# print('**COMMAND**')
# print(cmd)
# print('***********')


# os.system(cmd)


# #plot psths


