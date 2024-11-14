find $1 -name "*PointCloud*.xyz" | while read NAME ; do python adaps_to_common.py ${NAME} ; done
