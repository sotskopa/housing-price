??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint?
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
DenseBincount
input"Tidx
size"Tidx
weights"T
output"T"
Tidxtype:
2	"
Ttype:
2	"
binary_outputbool( 
=
Greater
x"T
y"T
z
"
Ttype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
$

LogicalAnd
x

y

z
?
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
:
Minimum
x"T
y"T
z"T"
Ttype:

2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*	2.5.0-rc02v1.12.1-53831-ga8b6d5ff93a8??
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
|
dense_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_36/kernel
u
#dense_36/kernel/Read/ReadVariableOpReadVariableOpdense_36/kernel* 
_output_shapes
:
??*
dtype0
s
dense_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_36/bias
l
!dense_36/bias/Read/ReadVariableOpReadVariableOpdense_36/bias*
_output_shapes	
:?*
dtype0
{
dense_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@* 
shared_namedense_37/kernel
t
#dense_37/kernel/Read/ReadVariableOpReadVariableOpdense_37/kernel*
_output_shapes
:	?@*
dtype0
r
dense_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_37/bias
k
!dense_37/bias/Read/ReadVariableOpReadVariableOpdense_37/bias*
_output_shapes
:@*
dtype0
z
dense_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ * 
shared_namedense_38/kernel
s
#dense_38/kernel/Read/ReadVariableOpReadVariableOpdense_38/kernel*
_output_shapes

:@ *
dtype0
r
dense_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_38/bias
k
!dense_38/bias/Read/ReadVariableOpReadVariableOpdense_38/bias*
_output_shapes
: *
dtype0
z
dense_39/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_39/kernel
s
#dense_39/kernel/Read/ReadVariableOpReadVariableOpdense_39/kernel*
_output_shapes

: *
dtype0
r
dense_39/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_39/bias
k
!dense_39/bias/Read/ReadVariableOpReadVariableOpdense_39/bias*
_output_shapes
:*
dtype0
`
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean
Y
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
:*
dtype0
h
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance
a
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0	
?
string_lookup_9_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_49019*
value_dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_36/kernel/m
?
*Adam/dense_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_36/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_36/bias/m
z
(Adam/dense_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_36/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_37/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*'
shared_nameAdam/dense_37/kernel/m
?
*Adam/dense_37/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_37/kernel/m*
_output_shapes
:	?@*
dtype0
?
Adam/dense_37/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_37/bias/m
y
(Adam/dense_37/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_37/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_38/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/dense_38/kernel/m
?
*Adam/dense_38/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_38/kernel/m*
_output_shapes

:@ *
dtype0
?
Adam/dense_38/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_38/bias/m
y
(Adam/dense_38/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_38/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_39/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_39/kernel/m
?
*Adam/dense_39/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_39/kernel/m*
_output_shapes

: *
dtype0
?
Adam/dense_39/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_39/bias/m
y
(Adam/dense_39/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_39/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_36/kernel/v
?
*Adam/dense_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_36/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_36/bias/v
z
(Adam/dense_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_36/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_37/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*'
shared_nameAdam/dense_37/kernel/v
?
*Adam/dense_37/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_37/kernel/v*
_output_shapes
:	?@*
dtype0
?
Adam/dense_37/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_37/bias/v
y
(Adam/dense_37/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_37/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_38/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/dense_38/kernel/v
?
*Adam/dense_38/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_38/kernel/v*
_output_shapes

:@ *
dtype0
?
Adam/dense_38/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_38/bias/v
y
(Adam/dense_38/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_38/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_39/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_39/kernel/v
?
*Adam/dense_39/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_39/kernel/v*
_output_shapes

: *
dtype0
?
Adam/dense_39/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_39/bias/v
y
(Adam/dense_39/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_39/bias/v*
_output_shapes
:*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *#
fR
__inference_<lambda>_53526

NoOpNoOp^PartitionedCall
?
Jstring_lookup_9_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2string_lookup_9_index_table*
Tkeys0*
Tvalues0	*.
_class$
" loc:@string_lookup_9_index_table*
_output_shapes

::
?=
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*?<
value?<B?< B?<
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-0
layer-7
	layer_with_weights-1
	layer-8

	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
 
 
 
 
 
 
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
layer_with_weights-0
layer-8
layer_with_weights-1
layer-9
layer-10
layer-11
trainable_variables
regularization_losses
	variables
	keras_api
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
trainable_variables
regularization_losses
	variables
 	keras_api
?
!iter

"beta_1

#beta_2
	$decay
%learning_rate&m?'m?(m?)m?*m?+m?,m?-m?&v?'v?(v?)v?*v?+v?,v?-v?
8
&0
'1
(2
)3
*4
+5
,6
-7
 
O
.1
/2
03
&4
'5
(6
)7
*8
+9
,10
-11
?
1metrics
trainable_variables
regularization_losses
2layer_regularization_losses
	variables
3non_trainable_variables

4layers
5layer_metrics
 
R
6trainable_variables
7regularization_losses
8	variables
9	keras_api
0
:state_variables

;_table
<	keras_api
?
=
_keep_axis
>_reduce_axis
?_reduce_axis_mask
@_broadcast_shape
.mean
/variance
	0count
A	keras_api

B	keras_api
R
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
 
 

.1
/2
03
?
Gmetrics
trainable_variables
regularization_losses
Hlayer_regularization_losses
	variables
Inon_trainable_variables

Jlayers
Klayer_metrics
h

&kernel
'bias
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
h

(kernel
)bias
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
h

*kernel
+bias
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
h

,kernel
-bias
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
8
&0
'1
(2
)3
*4
+5
,6
-7
 
8
&0
'1
(2
)3
*4
+5
,6
-7
?
\metrics
trainable_variables
regularization_losses
]layer_regularization_losses
	variables
^non_trainable_variables

_layers
`layer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_36/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_36/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_37/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_37/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_38/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_38/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_39/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_39/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
@>
VARIABLE_VALUEmean&variables/1/.ATTRIBUTES/VARIABLE_VALUE
DB
VARIABLE_VALUEvariance&variables/2/.ATTRIBUTES/VARIABLE_VALUE
A?
VARIABLE_VALUEcount&variables/3/.ATTRIBUTES/VARIABLE_VALUE

a0
 

.1
/2
03
?
0
1
2
3
4
5
6
7
	8
 
 
 
 
?
bmetrics
6trainable_variables
7regularization_losses
clayer_regularization_losses
8	variables
dnon_trainable_variables

elayers
flayer_metrics
 
MK
tableBlayer_with_weights-0/layer_with_weights-0/_table/.ATTRIBUTES/table
 
 
 
 
 
 
 
 
 
 
?
gmetrics
Ctrainable_variables
Dregularization_losses
hlayer_regularization_losses
E	variables
inon_trainable_variables

jlayers
klayer_metrics
 
 

.1
/2
03
V
0
1
2
3
4
5
6
7
8
9
10
11
 

&0
'1
 

&0
'1
?
lmetrics
Ltrainable_variables
Mregularization_losses
mlayer_regularization_losses
N	variables
nnon_trainable_variables

olayers
player_metrics

(0
)1
 

(0
)1
?
qmetrics
Ptrainable_variables
Qregularization_losses
rlayer_regularization_losses
R	variables
snon_trainable_variables

tlayers
ulayer_metrics

*0
+1
 

*0
+1
?
vmetrics
Ttrainable_variables
Uregularization_losses
wlayer_regularization_losses
V	variables
xnon_trainable_variables

ylayers
zlayer_metrics

,0
-1
 

,0
-1
?
{metrics
Xtrainable_variables
Yregularization_losses
|layer_regularization_losses
Z	variables
}non_trainable_variables

~layers
layer_metrics
 
 
 

0
1
2
3
 
8

?total

?count
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
xv
VARIABLE_VALUEAdam/dense_36/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_36/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_37/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_37/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_38/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_38/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_39/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_39/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_36/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_36/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_37/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_37/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_38/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_38/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_39/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_39/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
w
serving_default_areaPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
~
serving_default_garden_areaPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
}
serving_default_month_soldPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
x
serving_default_roomsPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????

serving_default_square_meterPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
$serving_default_year_of_constructionPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
|
serving_default_year_soldPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_areaserving_default_garden_areaserving_default_month_soldserving_default_roomsserving_default_square_meter$serving_default_year_of_constructionserving_default_year_soldstring_lookup_9_index_tableConstmeanvariancedense_36/kerneldense_36/biasdense_37/kerneldense_37/biasdense_38/kerneldense_38/biasdense_39/kerneldense_39/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_52824
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp#dense_36/kernel/Read/ReadVariableOp!dense_36/bias/Read/ReadVariableOp#dense_37/kernel/Read/ReadVariableOp!dense_37/bias/Read/ReadVariableOp#dense_38/kernel/Read/ReadVariableOp!dense_38/bias/Read/ReadVariableOp#dense_39/kernel/Read/ReadVariableOp!dense_39/bias/Read/ReadVariableOpmean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOpJstring_lookup_9_index_table_lookup_table_export_values/LookupTableExportV2Lstring_lookup_9_index_table_lookup_table_export_values/LookupTableExportV2:1total/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_36/kernel/m/Read/ReadVariableOp(Adam/dense_36/bias/m/Read/ReadVariableOp*Adam/dense_37/kernel/m/Read/ReadVariableOp(Adam/dense_37/bias/m/Read/ReadVariableOp*Adam/dense_38/kernel/m/Read/ReadVariableOp(Adam/dense_38/bias/m/Read/ReadVariableOp*Adam/dense_39/kernel/m/Read/ReadVariableOp(Adam/dense_39/bias/m/Read/ReadVariableOp*Adam/dense_36/kernel/v/Read/ReadVariableOp(Adam/dense_36/bias/v/Read/ReadVariableOp*Adam/dense_37/kernel/v/Read/ReadVariableOp(Adam/dense_37/bias/v/Read/ReadVariableOp*Adam/dense_38/kernel/v/Read/ReadVariableOp(Adam/dense_38/bias/v/Read/ReadVariableOp*Adam/dense_39/kernel/v/Read/ReadVariableOp(Adam/dense_39/bias/v/Read/ReadVariableOpConst_1*1
Tin*
(2&			*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_53664
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratedense_36/kerneldense_36/biasdense_37/kerneldense_37/biasdense_38/kerneldense_38/biasdense_39/kerneldense_39/biasmeanvariancecountstring_lookup_9_index_tabletotalcount_1Adam/dense_36/kernel/mAdam/dense_36/bias/mAdam/dense_37/kernel/mAdam/dense_37/bias/mAdam/dense_38/kernel/mAdam/dense_38/bias/mAdam/dense_39/kernel/mAdam/dense_39/bias/mAdam/dense_36/kernel/vAdam/dense_36/bias/vAdam/dense_37/kernel/vAdam/dense_37/bias/vAdam/dense_38/kernel/vAdam/dense_38/bias/vAdam/dense_39/kernel/vAdam/dense_39/bias/v*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_53779??
??
?
 __inference__wrapped_model_51880
area
garden_area

month_sold	
rooms
square_meter
year_of_construction
	year_sold[
Wmodel_19_model_18_string_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handle\
Xmodel_19_model_18_string_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value	O
Amodel_19_model_18_normalization_9_reshape_readvariableop_resource:Q
Cmodel_19_model_18_normalization_9_reshape_1_readvariableop_resource:Q
=model_19_sequential_9_dense_36_matmul_readvariableop_resource:
??M
>model_19_sequential_9_dense_36_biasadd_readvariableop_resource:	?P
=model_19_sequential_9_dense_37_matmul_readvariableop_resource:	?@L
>model_19_sequential_9_dense_37_biasadd_readvariableop_resource:@O
=model_19_sequential_9_dense_38_matmul_readvariableop_resource:@ L
>model_19_sequential_9_dense_38_biasadd_readvariableop_resource: O
=model_19_sequential_9_dense_39_matmul_readvariableop_resource: L
>model_19_sequential_9_dense_39_biasadd_readvariableop_resource:
identity??3model_19/model_18/category_encoding_9/Assert/Assert?8model_19/model_18/normalization_9/Reshape/ReadVariableOp?:model_19/model_18/normalization_9/Reshape_1/ReadVariableOp?Jmodel_19/model_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2?5model_19/sequential_9/dense_36/BiasAdd/ReadVariableOp?4model_19/sequential_9/dense_36/MatMul/ReadVariableOp?5model_19/sequential_9/dense_37/BiasAdd/ReadVariableOp?4model_19/sequential_9/dense_37/MatMul/ReadVariableOp?5model_19/sequential_9/dense_38/BiasAdd/ReadVariableOp?4model_19/sequential_9/dense_38/MatMul/ReadVariableOp?5model_19/sequential_9/dense_39/BiasAdd/ReadVariableOp?4model_19/sequential_9/dense_39/MatMul/ReadVariableOp?
Jmodel_19/model_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Wmodel_19_model_18_string_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleareaXmodel_19_model_18_string_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2L
Jmodel_19/model_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2?
,model_19/model_18/concatenate_18/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2.
,model_19/model_18/concatenate_18/concat/axis?
'model_19/model_18/concatenate_18/concatConcatV2roomssquare_metergarden_areayear_of_construction	year_sold
month_sold5model_19/model_18/concatenate_18/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2)
'model_19/model_18/concatenate_18/concat?
8model_19/model_18/normalization_9/Reshape/ReadVariableOpReadVariableOpAmodel_19_model_18_normalization_9_reshape_readvariableop_resource*
_output_shapes
:*
dtype02:
8model_19/model_18/normalization_9/Reshape/ReadVariableOp?
/model_19/model_18/normalization_9/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      21
/model_19/model_18/normalization_9/Reshape/shape?
)model_19/model_18/normalization_9/ReshapeReshape@model_19/model_18/normalization_9/Reshape/ReadVariableOp:value:08model_19/model_18/normalization_9/Reshape/shape:output:0*
T0*
_output_shapes

:2+
)model_19/model_18/normalization_9/Reshape?
:model_19/model_18/normalization_9/Reshape_1/ReadVariableOpReadVariableOpCmodel_19_model_18_normalization_9_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02<
:model_19/model_18/normalization_9/Reshape_1/ReadVariableOp?
1model_19/model_18/normalization_9/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      23
1model_19/model_18/normalization_9/Reshape_1/shape?
+model_19/model_18/normalization_9/Reshape_1ReshapeBmodel_19/model_18/normalization_9/Reshape_1/ReadVariableOp:value:0:model_19/model_18/normalization_9/Reshape_1/shape:output:0*
T0*
_output_shapes

:2-
+model_19/model_18/normalization_9/Reshape_1?
%model_19/model_18/normalization_9/subSub0model_19/model_18/concatenate_18/concat:output:02model_19/model_18/normalization_9/Reshape:output:0*
T0*'
_output_shapes
:?????????2'
%model_19/model_18/normalization_9/sub?
&model_19/model_18/normalization_9/SqrtSqrt4model_19/model_18/normalization_9/Reshape_1:output:0*
T0*
_output_shapes

:2(
&model_19/model_18/normalization_9/Sqrt?
+model_19/model_18/normalization_9/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32-
+model_19/model_18/normalization_9/Maximum/y?
)model_19/model_18/normalization_9/MaximumMaximum*model_19/model_18/normalization_9/Sqrt:y:04model_19/model_18/normalization_9/Maximum/y:output:0*
T0*
_output_shapes

:2+
)model_19/model_18/normalization_9/Maximum?
)model_19/model_18/normalization_9/truedivRealDiv)model_19/model_18/normalization_9/sub:z:0-model_19/model_18/normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????2+
)model_19/model_18/normalization_9/truediv?
+model_19/model_18/category_encoding_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+model_19/model_18/category_encoding_9/Const?
)model_19/model_18/category_encoding_9/MaxMaxSmodel_19/model_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2:values:04model_19/model_18/category_encoding_9/Const:output:0*
T0	*
_output_shapes
: 2+
)model_19/model_18/category_encoding_9/Max?
-model_19/model_18/category_encoding_9/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-model_19/model_18/category_encoding_9/Const_1?
)model_19/model_18/category_encoding_9/MinMinSmodel_19/model_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2:values:06model_19/model_18/category_encoding_9/Const_1:output:0*
T0	*
_output_shapes
: 2+
)model_19/model_18/category_encoding_9/Min?
,model_19/model_18/category_encoding_9/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2.
,model_19/model_18/category_encoding_9/Cast/x?
*model_19/model_18/category_encoding_9/CastCast5model_19/model_18/category_encoding_9/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2,
*model_19/model_18/category_encoding_9/Cast?
-model_19/model_18/category_encoding_9/GreaterGreater.model_19/model_18/category_encoding_9/Cast:y:02model_19/model_18/category_encoding_9/Max:output:0*
T0	*
_output_shapes
: 2/
-model_19/model_18/category_encoding_9/Greater?
.model_19/model_18/category_encoding_9/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 20
.model_19/model_18/category_encoding_9/Cast_1/x?
,model_19/model_18/category_encoding_9/Cast_1Cast7model_19/model_18/category_encoding_9/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2.
,model_19/model_18/category_encoding_9/Cast_1?
2model_19/model_18/category_encoding_9/GreaterEqualGreaterEqual2model_19/model_18/category_encoding_9/Min:output:00model_19/model_18/category_encoding_9/Cast_1:y:0*
T0	*
_output_shapes
: 24
2model_19/model_18/category_encoding_9/GreaterEqual?
0model_19/model_18/category_encoding_9/LogicalAnd
LogicalAnd1model_19/model_18/category_encoding_9/Greater:z:06model_19/model_18/category_encoding_9/GreaterEqual:z:0*
_output_shapes
: 22
0model_19/model_18/category_encoding_9/LogicalAnd?
2model_19/model_18/category_encoding_9/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=26224
2model_19/model_18/category_encoding_9/Assert/Const?
:model_19/model_18/category_encoding_9/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2622<
:model_19/model_18/category_encoding_9/Assert/Assert/data_0?
3model_19/model_18/category_encoding_9/Assert/AssertAssert4model_19/model_18/category_encoding_9/LogicalAnd:z:0Cmodel_19/model_18/category_encoding_9/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 25
3model_19/model_18/category_encoding_9/Assert/Assert?
4model_19/model_18/category_encoding_9/bincount/ShapeShapeSmodel_19/model_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:26
4model_19/model_18/category_encoding_9/bincount/Shape?
4model_19/model_18/category_encoding_9/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4model_19/model_18/category_encoding_9/bincount/Const?
3model_19/model_18/category_encoding_9/bincount/ProdProd=model_19/model_18/category_encoding_9/bincount/Shape:output:0=model_19/model_18/category_encoding_9/bincount/Const:output:0*
T0*
_output_shapes
: 25
3model_19/model_18/category_encoding_9/bincount/Prod?
8model_19/model_18/category_encoding_9/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2:
8model_19/model_18/category_encoding_9/bincount/Greater/y?
6model_19/model_18/category_encoding_9/bincount/GreaterGreater<model_19/model_18/category_encoding_9/bincount/Prod:output:0Amodel_19/model_18/category_encoding_9/bincount/Greater/y:output:0*
T0*
_output_shapes
: 28
6model_19/model_18/category_encoding_9/bincount/Greater?
3model_19/model_18/category_encoding_9/bincount/CastCast:model_19/model_18/category_encoding_9/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 25
3model_19/model_18/category_encoding_9/bincount/Cast?
6model_19/model_18/category_encoding_9/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       28
6model_19/model_18/category_encoding_9/bincount/Const_1?
2model_19/model_18/category_encoding_9/bincount/MaxMaxSmodel_19/model_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0?model_19/model_18/category_encoding_9/bincount/Const_1:output:0*
T0	*
_output_shapes
: 24
2model_19/model_18/category_encoding_9/bincount/Max?
4model_19/model_18/category_encoding_9/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R26
4model_19/model_18/category_encoding_9/bincount/add/y?
2model_19/model_18/category_encoding_9/bincount/addAddV2;model_19/model_18/category_encoding_9/bincount/Max:output:0=model_19/model_18/category_encoding_9/bincount/add/y:output:0*
T0	*
_output_shapes
: 24
2model_19/model_18/category_encoding_9/bincount/add?
2model_19/model_18/category_encoding_9/bincount/mulMul7model_19/model_18/category_encoding_9/bincount/Cast:y:06model_19/model_18/category_encoding_9/bincount/add:z:0*
T0	*
_output_shapes
: 24
2model_19/model_18/category_encoding_9/bincount/mul?
8model_19/model_18/category_encoding_9/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2:
8model_19/model_18/category_encoding_9/bincount/minlength?
6model_19/model_18/category_encoding_9/bincount/MaximumMaximumAmodel_19/model_18/category_encoding_9/bincount/minlength:output:06model_19/model_18/category_encoding_9/bincount/mul:z:0*
T0	*
_output_shapes
: 28
6model_19/model_18/category_encoding_9/bincount/Maximum?
8model_19/model_18/category_encoding_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2:
8model_19/model_18/category_encoding_9/bincount/maxlength?
6model_19/model_18/category_encoding_9/bincount/MinimumMinimumAmodel_19/model_18/category_encoding_9/bincount/maxlength:output:0:model_19/model_18/category_encoding_9/bincount/Maximum:z:0*
T0	*
_output_shapes
: 28
6model_19/model_18/category_encoding_9/bincount/Minimum?
6model_19/model_18/category_encoding_9/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 28
6model_19/model_18/category_encoding_9/bincount/Const_2?
<model_19/model_18/category_encoding_9/bincount/DenseBincountDenseBincountSmodel_19/model_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0:model_19/model_18/category_encoding_9/bincount/Minimum:z:0?model_19/model_18/category_encoding_9/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2>
<model_19/model_18/category_encoding_9/bincount/DenseBincount?
,model_19/model_18/concatenate_19/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2.
,model_19/model_18/concatenate_19/concat/axis?
'model_19/model_18/concatenate_19/concatConcatV2-model_19/model_18/normalization_9/truediv:z:0Emodel_19/model_18/category_encoding_9/bincount/DenseBincount:output:05model_19/model_18/concatenate_19/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2)
'model_19/model_18/concatenate_19/concat?
4model_19/sequential_9/dense_36/MatMul/ReadVariableOpReadVariableOp=model_19_sequential_9_dense_36_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype026
4model_19/sequential_9/dense_36/MatMul/ReadVariableOp?
%model_19/sequential_9/dense_36/MatMulMatMul0model_19/model_18/concatenate_19/concat:output:0<model_19/sequential_9/dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%model_19/sequential_9/dense_36/MatMul?
5model_19/sequential_9/dense_36/BiasAdd/ReadVariableOpReadVariableOp>model_19_sequential_9_dense_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype027
5model_19/sequential_9/dense_36/BiasAdd/ReadVariableOp?
&model_19/sequential_9/dense_36/BiasAddBiasAdd/model_19/sequential_9/dense_36/MatMul:product:0=model_19/sequential_9/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&model_19/sequential_9/dense_36/BiasAdd?
#model_19/sequential_9/dense_36/TanhTanh/model_19/sequential_9/dense_36/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2%
#model_19/sequential_9/dense_36/Tanh?
4model_19/sequential_9/dense_37/MatMul/ReadVariableOpReadVariableOp=model_19_sequential_9_dense_37_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype026
4model_19/sequential_9/dense_37/MatMul/ReadVariableOp?
%model_19/sequential_9/dense_37/MatMulMatMul'model_19/sequential_9/dense_36/Tanh:y:0<model_19/sequential_9/dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2'
%model_19/sequential_9/dense_37/MatMul?
5model_19/sequential_9/dense_37/BiasAdd/ReadVariableOpReadVariableOp>model_19_sequential_9_dense_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype027
5model_19/sequential_9/dense_37/BiasAdd/ReadVariableOp?
&model_19/sequential_9/dense_37/BiasAddBiasAdd/model_19/sequential_9/dense_37/MatMul:product:0=model_19/sequential_9/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2(
&model_19/sequential_9/dense_37/BiasAdd?
4model_19/sequential_9/dense_38/MatMul/ReadVariableOpReadVariableOp=model_19_sequential_9_dense_38_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype026
4model_19/sequential_9/dense_38/MatMul/ReadVariableOp?
%model_19/sequential_9/dense_38/MatMulMatMul/model_19/sequential_9/dense_37/BiasAdd:output:0<model_19/sequential_9/dense_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2'
%model_19/sequential_9/dense_38/MatMul?
5model_19/sequential_9/dense_38/BiasAdd/ReadVariableOpReadVariableOp>model_19_sequential_9_dense_38_biasadd_readvariableop_resource*
_output_shapes
: *
dtype027
5model_19/sequential_9/dense_38/BiasAdd/ReadVariableOp?
&model_19/sequential_9/dense_38/BiasAddBiasAdd/model_19/sequential_9/dense_38/MatMul:product:0=model_19/sequential_9/dense_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&model_19/sequential_9/dense_38/BiasAdd?
4model_19/sequential_9/dense_39/MatMul/ReadVariableOpReadVariableOp=model_19_sequential_9_dense_39_matmul_readvariableop_resource*
_output_shapes

: *
dtype026
4model_19/sequential_9/dense_39/MatMul/ReadVariableOp?
%model_19/sequential_9/dense_39/MatMulMatMul/model_19/sequential_9/dense_38/BiasAdd:output:0<model_19/sequential_9/dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2'
%model_19/sequential_9/dense_39/MatMul?
5model_19/sequential_9/dense_39/BiasAdd/ReadVariableOpReadVariableOp>model_19_sequential_9_dense_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5model_19/sequential_9/dense_39/BiasAdd/ReadVariableOp?
&model_19/sequential_9/dense_39/BiasAddBiasAdd/model_19/sequential_9/dense_39/MatMul:product:0=model_19/sequential_9/dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&model_19/sequential_9/dense_39/BiasAdd?
IdentityIdentity/model_19/sequential_9/dense_39/BiasAdd:output:04^model_19/model_18/category_encoding_9/Assert/Assert9^model_19/model_18/normalization_9/Reshape/ReadVariableOp;^model_19/model_18/normalization_9/Reshape_1/ReadVariableOpK^model_19/model_18/string_lookup_9/None_lookup_table_find/LookupTableFindV26^model_19/sequential_9/dense_36/BiasAdd/ReadVariableOp5^model_19/sequential_9/dense_36/MatMul/ReadVariableOp6^model_19/sequential_9/dense_37/BiasAdd/ReadVariableOp5^model_19/sequential_9/dense_37/MatMul/ReadVariableOp6^model_19/sequential_9/dense_38/BiasAdd/ReadVariableOp5^model_19/sequential_9/dense_38/MatMul/ReadVariableOp6^model_19/sequential_9/dense_39/BiasAdd/ReadVariableOp5^model_19/sequential_9/dense_39/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2j
3model_19/model_18/category_encoding_9/Assert/Assert3model_19/model_18/category_encoding_9/Assert/Assert2t
8model_19/model_18/normalization_9/Reshape/ReadVariableOp8model_19/model_18/normalization_9/Reshape/ReadVariableOp2x
:model_19/model_18/normalization_9/Reshape_1/ReadVariableOp:model_19/model_18/normalization_9/Reshape_1/ReadVariableOp2?
Jmodel_19/model_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2Jmodel_19/model_18/string_lookup_9/None_lookup_table_find/LookupTableFindV22n
5model_19/sequential_9/dense_36/BiasAdd/ReadVariableOp5model_19/sequential_9/dense_36/BiasAdd/ReadVariableOp2l
4model_19/sequential_9/dense_36/MatMul/ReadVariableOp4model_19/sequential_9/dense_36/MatMul/ReadVariableOp2n
5model_19/sequential_9/dense_37/BiasAdd/ReadVariableOp5model_19/sequential_9/dense_37/BiasAdd/ReadVariableOp2l
4model_19/sequential_9/dense_37/MatMul/ReadVariableOp4model_19/sequential_9/dense_37/MatMul/ReadVariableOp2n
5model_19/sequential_9/dense_38/BiasAdd/ReadVariableOp5model_19/sequential_9/dense_38/BiasAdd/ReadVariableOp2l
4model_19/sequential_9/dense_38/MatMul/ReadVariableOp4model_19/sequential_9/dense_38/MatMul/ReadVariableOp2n
5model_19/sequential_9/dense_39/BiasAdd/ReadVariableOp5model_19/sequential_9/dense_39/BiasAdd/ReadVariableOp2l
4model_19/sequential_9/dense_39/MatMul/ReadVariableOp4model_19/sequential_9/dense_39/MatMul/ReadVariableOp:M I
'
_output_shapes
:?????????

_user_specified_namearea:TP
'
_output_shapes
:?????????
%
_user_specified_namegarden_area:SO
'
_output_shapes
:?????????
$
_user_specified_name
month_sold:NJ
'
_output_shapes
:?????????

_user_specified_namerooms:UQ
'
_output_shapes
:?????????
&
_user_specified_namesquare_meter:]Y
'
_output_shapes
:?????????
.
_user_specified_nameyear_of_construction:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:

_output_shapes
: 
?
Z
.__inference_concatenate_19_layer_call_fn_53402
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_19_layer_call_and_return_conditional_losses_519652
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????:??????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
??
?
C__inference_model_19_layer_call_and_return_conditional_losses_52909
inputs_area
inputs_garden_area
inputs_month_sold
inputs_rooms
inputs_square_meter
inputs_year_of_construction
inputs_year_soldR
Nmodel_18_string_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleS
Omodel_18_string_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value	F
8model_18_normalization_9_reshape_readvariableop_resource:H
:model_18_normalization_9_reshape_1_readvariableop_resource:H
4sequential_9_dense_36_matmul_readvariableop_resource:
??D
5sequential_9_dense_36_biasadd_readvariableop_resource:	?G
4sequential_9_dense_37_matmul_readvariableop_resource:	?@C
5sequential_9_dense_37_biasadd_readvariableop_resource:@F
4sequential_9_dense_38_matmul_readvariableop_resource:@ C
5sequential_9_dense_38_biasadd_readvariableop_resource: F
4sequential_9_dense_39_matmul_readvariableop_resource: C
5sequential_9_dense_39_biasadd_readvariableop_resource:
identity??*model_18/category_encoding_9/Assert/Assert?/model_18/normalization_9/Reshape/ReadVariableOp?1model_18/normalization_9/Reshape_1/ReadVariableOp?Amodel_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2?,sequential_9/dense_36/BiasAdd/ReadVariableOp?+sequential_9/dense_36/MatMul/ReadVariableOp?,sequential_9/dense_37/BiasAdd/ReadVariableOp?+sequential_9/dense_37/MatMul/ReadVariableOp?,sequential_9/dense_38/BiasAdd/ReadVariableOp?+sequential_9/dense_38/MatMul/ReadVariableOp?,sequential_9/dense_39/BiasAdd/ReadVariableOp?+sequential_9/dense_39/MatMul/ReadVariableOp?
Amodel_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Nmodel_18_string_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleinputs_areaOmodel_18_string_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2C
Amodel_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2?
#model_18/concatenate_18/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2%
#model_18/concatenate_18/concat/axis?
model_18/concatenate_18/concatConcatV2inputs_roomsinputs_square_meterinputs_garden_areainputs_year_of_constructioninputs_year_soldinputs_month_sold,model_18/concatenate_18/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2 
model_18/concatenate_18/concat?
/model_18/normalization_9/Reshape/ReadVariableOpReadVariableOp8model_18_normalization_9_reshape_readvariableop_resource*
_output_shapes
:*
dtype021
/model_18/normalization_9/Reshape/ReadVariableOp?
&model_18/normalization_9/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2(
&model_18/normalization_9/Reshape/shape?
 model_18/normalization_9/ReshapeReshape7model_18/normalization_9/Reshape/ReadVariableOp:value:0/model_18/normalization_9/Reshape/shape:output:0*
T0*
_output_shapes

:2"
 model_18/normalization_9/Reshape?
1model_18/normalization_9/Reshape_1/ReadVariableOpReadVariableOp:model_18_normalization_9_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype023
1model_18/normalization_9/Reshape_1/ReadVariableOp?
(model_18/normalization_9/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2*
(model_18/normalization_9/Reshape_1/shape?
"model_18/normalization_9/Reshape_1Reshape9model_18/normalization_9/Reshape_1/ReadVariableOp:value:01model_18/normalization_9/Reshape_1/shape:output:0*
T0*
_output_shapes

:2$
"model_18/normalization_9/Reshape_1?
model_18/normalization_9/subSub'model_18/concatenate_18/concat:output:0)model_18/normalization_9/Reshape:output:0*
T0*'
_output_shapes
:?????????2
model_18/normalization_9/sub?
model_18/normalization_9/SqrtSqrt+model_18/normalization_9/Reshape_1:output:0*
T0*
_output_shapes

:2
model_18/normalization_9/Sqrt?
"model_18/normalization_9/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32$
"model_18/normalization_9/Maximum/y?
 model_18/normalization_9/MaximumMaximum!model_18/normalization_9/Sqrt:y:0+model_18/normalization_9/Maximum/y:output:0*
T0*
_output_shapes

:2"
 model_18/normalization_9/Maximum?
 model_18/normalization_9/truedivRealDiv model_18/normalization_9/sub:z:0$model_18/normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????2"
 model_18/normalization_9/truediv?
"model_18/category_encoding_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model_18/category_encoding_9/Const?
 model_18/category_encoding_9/MaxMaxJmodel_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0+model_18/category_encoding_9/Const:output:0*
T0	*
_output_shapes
: 2"
 model_18/category_encoding_9/Max?
$model_18/category_encoding_9/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$model_18/category_encoding_9/Const_1?
 model_18/category_encoding_9/MinMinJmodel_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0-model_18/category_encoding_9/Const_1:output:0*
T0	*
_output_shapes
: 2"
 model_18/category_encoding_9/Min?
#model_18/category_encoding_9/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2%
#model_18/category_encoding_9/Cast/x?
!model_18/category_encoding_9/CastCast,model_18/category_encoding_9/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2#
!model_18/category_encoding_9/Cast?
$model_18/category_encoding_9/GreaterGreater%model_18/category_encoding_9/Cast:y:0)model_18/category_encoding_9/Max:output:0*
T0	*
_output_shapes
: 2&
$model_18/category_encoding_9/Greater?
%model_18/category_encoding_9/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2'
%model_18/category_encoding_9/Cast_1/x?
#model_18/category_encoding_9/Cast_1Cast.model_18/category_encoding_9/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2%
#model_18/category_encoding_9/Cast_1?
)model_18/category_encoding_9/GreaterEqualGreaterEqual)model_18/category_encoding_9/Min:output:0'model_18/category_encoding_9/Cast_1:y:0*
T0	*
_output_shapes
: 2+
)model_18/category_encoding_9/GreaterEqual?
'model_18/category_encoding_9/LogicalAnd
LogicalAnd(model_18/category_encoding_9/Greater:z:0-model_18/category_encoding_9/GreaterEqual:z:0*
_output_shapes
: 2)
'model_18/category_encoding_9/LogicalAnd?
)model_18/category_encoding_9/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2622+
)model_18/category_encoding_9/Assert/Const?
1model_18/category_encoding_9/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=26223
1model_18/category_encoding_9/Assert/Assert/data_0?
*model_18/category_encoding_9/Assert/AssertAssert+model_18/category_encoding_9/LogicalAnd:z:0:model_18/category_encoding_9/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2,
*model_18/category_encoding_9/Assert/Assert?
+model_18/category_encoding_9/bincount/ShapeShapeJmodel_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2-
+model_18/category_encoding_9/bincount/Shape?
+model_18/category_encoding_9/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+model_18/category_encoding_9/bincount/Const?
*model_18/category_encoding_9/bincount/ProdProd4model_18/category_encoding_9/bincount/Shape:output:04model_18/category_encoding_9/bincount/Const:output:0*
T0*
_output_shapes
: 2,
*model_18/category_encoding_9/bincount/Prod?
/model_18/category_encoding_9/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 21
/model_18/category_encoding_9/bincount/Greater/y?
-model_18/category_encoding_9/bincount/GreaterGreater3model_18/category_encoding_9/bincount/Prod:output:08model_18/category_encoding_9/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2/
-model_18/category_encoding_9/bincount/Greater?
*model_18/category_encoding_9/bincount/CastCast1model_18/category_encoding_9/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2,
*model_18/category_encoding_9/bincount/Cast?
-model_18/category_encoding_9/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-model_18/category_encoding_9/bincount/Const_1?
)model_18/category_encoding_9/bincount/MaxMaxJmodel_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2:values:06model_18/category_encoding_9/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2+
)model_18/category_encoding_9/bincount/Max?
+model_18/category_encoding_9/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2-
+model_18/category_encoding_9/bincount/add/y?
)model_18/category_encoding_9/bincount/addAddV22model_18/category_encoding_9/bincount/Max:output:04model_18/category_encoding_9/bincount/add/y:output:0*
T0	*
_output_shapes
: 2+
)model_18/category_encoding_9/bincount/add?
)model_18/category_encoding_9/bincount/mulMul.model_18/category_encoding_9/bincount/Cast:y:0-model_18/category_encoding_9/bincount/add:z:0*
T0	*
_output_shapes
: 2+
)model_18/category_encoding_9/bincount/mul?
/model_18/category_encoding_9/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?21
/model_18/category_encoding_9/bincount/minlength?
-model_18/category_encoding_9/bincount/MaximumMaximum8model_18/category_encoding_9/bincount/minlength:output:0-model_18/category_encoding_9/bincount/mul:z:0*
T0	*
_output_shapes
: 2/
-model_18/category_encoding_9/bincount/Maximum?
/model_18/category_encoding_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?21
/model_18/category_encoding_9/bincount/maxlength?
-model_18/category_encoding_9/bincount/MinimumMinimum8model_18/category_encoding_9/bincount/maxlength:output:01model_18/category_encoding_9/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2/
-model_18/category_encoding_9/bincount/Minimum?
-model_18/category_encoding_9/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2/
-model_18/category_encoding_9/bincount/Const_2?
3model_18/category_encoding_9/bincount/DenseBincountDenseBincountJmodel_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2:values:01model_18/category_encoding_9/bincount/Minimum:z:06model_18/category_encoding_9/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(25
3model_18/category_encoding_9/bincount/DenseBincount?
#model_18/concatenate_19/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2%
#model_18/concatenate_19/concat/axis?
model_18/concatenate_19/concatConcatV2$model_18/normalization_9/truediv:z:0<model_18/category_encoding_9/bincount/DenseBincount:output:0,model_18/concatenate_19/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2 
model_18/concatenate_19/concat?
+sequential_9/dense_36/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_36_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+sequential_9/dense_36/MatMul/ReadVariableOp?
sequential_9/dense_36/MatMulMatMul'model_18/concatenate_19/concat:output:03sequential_9/dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_36/MatMul?
,sequential_9/dense_36/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential_9/dense_36/BiasAdd/ReadVariableOp?
sequential_9/dense_36/BiasAddBiasAdd&sequential_9/dense_36/MatMul:product:04sequential_9/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_36/BiasAdd?
sequential_9/dense_36/TanhTanh&sequential_9/dense_36/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_36/Tanh?
+sequential_9/dense_37/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_37_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02-
+sequential_9/dense_37/MatMul/ReadVariableOp?
sequential_9/dense_37/MatMulMatMulsequential_9/dense_36/Tanh:y:03sequential_9/dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential_9/dense_37/MatMul?
,sequential_9/dense_37/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_9/dense_37/BiasAdd/ReadVariableOp?
sequential_9/dense_37/BiasAddBiasAdd&sequential_9/dense_37/MatMul:product:04sequential_9/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential_9/dense_37/BiasAdd?
+sequential_9/dense_38/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_38_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02-
+sequential_9/dense_38/MatMul/ReadVariableOp?
sequential_9/dense_38/MatMulMatMul&sequential_9/dense_37/BiasAdd:output:03sequential_9/dense_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_9/dense_38/MatMul?
,sequential_9/dense_38/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_38_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_9/dense_38/BiasAdd/ReadVariableOp?
sequential_9/dense_38/BiasAddBiasAdd&sequential_9/dense_38/MatMul:product:04sequential_9/dense_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_9/dense_38/BiasAdd?
+sequential_9/dense_39/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_39_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+sequential_9/dense_39/MatMul/ReadVariableOp?
sequential_9/dense_39/MatMulMatMul&sequential_9/dense_38/BiasAdd:output:03sequential_9/dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_9/dense_39/MatMul?
,sequential_9/dense_39/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_9/dense_39/BiasAdd/ReadVariableOp?
sequential_9/dense_39/BiasAddBiasAdd&sequential_9/dense_39/MatMul:product:04sequential_9/dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_9/dense_39/BiasAdd?
IdentityIdentity&sequential_9/dense_39/BiasAdd:output:0+^model_18/category_encoding_9/Assert/Assert0^model_18/normalization_9/Reshape/ReadVariableOp2^model_18/normalization_9/Reshape_1/ReadVariableOpB^model_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2-^sequential_9/dense_36/BiasAdd/ReadVariableOp,^sequential_9/dense_36/MatMul/ReadVariableOp-^sequential_9/dense_37/BiasAdd/ReadVariableOp,^sequential_9/dense_37/MatMul/ReadVariableOp-^sequential_9/dense_38/BiasAdd/ReadVariableOp,^sequential_9/dense_38/MatMul/ReadVariableOp-^sequential_9/dense_39/BiasAdd/ReadVariableOp,^sequential_9/dense_39/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2X
*model_18/category_encoding_9/Assert/Assert*model_18/category_encoding_9/Assert/Assert2b
/model_18/normalization_9/Reshape/ReadVariableOp/model_18/normalization_9/Reshape/ReadVariableOp2f
1model_18/normalization_9/Reshape_1/ReadVariableOp1model_18/normalization_9/Reshape_1/ReadVariableOp2?
Amodel_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2Amodel_18/string_lookup_9/None_lookup_table_find/LookupTableFindV22\
,sequential_9/dense_36/BiasAdd/ReadVariableOp,sequential_9/dense_36/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_36/MatMul/ReadVariableOp+sequential_9/dense_36/MatMul/ReadVariableOp2\
,sequential_9/dense_37/BiasAdd/ReadVariableOp,sequential_9/dense_37/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_37/MatMul/ReadVariableOp+sequential_9/dense_37/MatMul/ReadVariableOp2\
,sequential_9/dense_38/BiasAdd/ReadVariableOp,sequential_9/dense_38/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_38/MatMul/ReadVariableOp+sequential_9/dense_38/MatMul/ReadVariableOp2\
,sequential_9/dense_39/BiasAdd/ReadVariableOp,sequential_9/dense_39/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_39/MatMul/ReadVariableOp+sequential_9/dense_39/MatMul/ReadVariableOp:T P
'
_output_shapes
:?????????
%
_user_specified_nameinputs/area:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/garden_area:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/month_sold:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/rooms:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/square_meter:d`
'
_output_shapes
:?????????
5
_user_specified_nameinputs/year_of_construction:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/year_sold:

_output_shapes
: 
?	
?
C__inference_dense_39_layer_call_and_return_conditional_losses_53470

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
C__inference_dense_37_layer_call_and_return_conditional_losses_52262

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?,
?
__inference_adapt_step_53389
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
22
IteratorGetNext?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeanIteratorGetNext:components:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceIteratorGetNext:components:0moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1j
ShapeShapeIteratorGetNext:components:0*
T0*
_output_shapes
:*
out_type0	2
Shapen
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:2

GatherV2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstX
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: 2
Prod|
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	2
add/ReadVariableOp_
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: 2
addS
CastCastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2
CastQ
Cast_1Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_1T
truedivRealDivCast:y:0
Cast_1:y:0*
T0*
_output_shapes
: 2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xO
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: 2
subt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpW
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:2
mula
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:2
mul_1P
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:2
add_1x
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1_
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:2
sub_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yQ
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:2
powz
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype02
ReadVariableOp_2_
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:2
add_2N
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:2
mul_2_
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:2
sub_2W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yW
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:2
pow_1c
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:2
add_3R
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:2
mul_3R
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:2
add_4?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignVariableOp?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype02
AssignVariableOp_1?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOp_2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?
.
__inference__initializer_53489
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_52407

inputs"
dense_36_52386:
??
dense_36_52388:	?!
dense_37_52391:	?@
dense_37_52393:@ 
dense_38_52396:@ 
dense_38_52398:  
dense_39_52401: 
dense_39_52403:
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall? dense_38/StatefulPartitionedCall? dense_39/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCallinputsdense_36_52386dense_36_52388*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_36_layer_call_and_return_conditional_losses_522462"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_52391dense_37_52393*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_37_layer_call_and_return_conditional_losses_522622"
 dense_37/StatefulPartitionedCall?
 dense_38/StatefulPartitionedCallStatefulPartitionedCall)dense_37/StatefulPartitionedCall:output:0dense_38_52396dense_38_52398*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_38_layer_call_and_return_conditional_losses_522782"
 dense_38/StatefulPartitionedCall?
 dense_39/StatefulPartitionedCallStatefulPartitionedCall)dense_38/StatefulPartitionedCall:output:0dense_39_52401dense_39_52403*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_39_layer_call_and_return_conditional_losses_522942"
 dense_39/StatefulPartitionedCall?
IdentityIdentity)dense_39/StatefulPartitionedCall:output:0!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_36_layer_call_fn_53422

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_36_layer_call_and_return_conditional_losses_522462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_model_19_layer_call_fn_53064
inputs_area
inputs_garden_area
inputs_month_sold
inputs_rooms
inputs_square_meter
inputs_year_of_construction
inputs_year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_areainputs_garden_areainputs_month_soldinputs_roomsinputs_square_meterinputs_year_of_constructioninputs_year_soldunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_19_layer_call_and_return_conditional_losses_526472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_nameinputs/area:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/garden_area:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/month_sold:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/rooms:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/square_meter:d`
'
_output_shapes
:?????????
5
_user_specified_nameinputs/year_of_construction:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/year_sold:

_output_shapes
: 
?	
?
,__inference_sequential_9_layer_call_fn_53301

inputs
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_523012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_37_layer_call_and_return_conditional_losses_53432

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
s
I__inference_concatenate_19_layer_call_and_return_conditional_losses_51965

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????:??????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_36_layer_call_and_return_conditional_losses_52246

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?T
?
C__inference_model_18_layer_call_and_return_conditional_losses_51968

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6I
Estring_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value	=
/normalization_9_reshape_readvariableop_resource:?
1normalization_9_reshape_1_readvariableop_resource:
identity??!category_encoding_9/Assert/Assert?&normalization_9/Reshape/ReadVariableOp?(normalization_9/Reshape_1/ReadVariableOp?8string_lookup_9/None_lookup_table_find/LookupTableFindV2?
8string_lookup_9/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleinputsFstring_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_9/None_lookup_table_find/LookupTableFindV2?
concatenate_18/PartitionedCallPartitionedCallinputs_3inputs_4inputs_1inputs_5inputs_6inputs_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_18_layer_call_and_return_conditional_losses_519132 
concatenate_18/PartitionedCall?
&normalization_9/Reshape/ReadVariableOpReadVariableOp/normalization_9_reshape_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization_9/Reshape/ReadVariableOp?
normalization_9/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_9/Reshape/shape?
normalization_9/ReshapeReshape.normalization_9/Reshape/ReadVariableOp:value:0&normalization_9/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_9/Reshape?
(normalization_9/Reshape_1/ReadVariableOpReadVariableOp1normalization_9_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02*
(normalization_9/Reshape_1/ReadVariableOp?
normalization_9/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_9/Reshape_1/shape?
normalization_9/Reshape_1Reshape0normalization_9/Reshape_1/ReadVariableOp:value:0(normalization_9/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_9/Reshape_1?
normalization_9/subSub'concatenate_18/PartitionedCall:output:0 normalization_9/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization_9/sub?
normalization_9/SqrtSqrt"normalization_9/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_9/Sqrt{
normalization_9/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_9/Maximum/y?
normalization_9/MaximumMaximumnormalization_9/Sqrt:y:0"normalization_9/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_9/Maximum?
normalization_9/truedivRealDivnormalization_9/sub:z:0normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_9/truediv?
category_encoding_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_9/Const?
category_encoding_9/MaxMaxAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0"category_encoding_9/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_9/Max?
category_encoding_9/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_9/Const_1?
category_encoding_9/MinMinAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0$category_encoding_9/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_9/Min{
category_encoding_9/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding_9/Cast/x?
category_encoding_9/CastCast#category_encoding_9/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_9/Cast?
category_encoding_9/GreaterGreatercategory_encoding_9/Cast:y:0 category_encoding_9/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_9/Greater~
category_encoding_9/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_9/Cast_1/x?
category_encoding_9/Cast_1Cast%category_encoding_9/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_9/Cast_1?
 category_encoding_9/GreaterEqualGreaterEqual category_encoding_9/Min:output:0category_encoding_9/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_9/GreaterEqual?
category_encoding_9/LogicalAnd
LogicalAndcategory_encoding_9/Greater:z:0$category_encoding_9/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_9/LogicalAnd?
 category_encoding_9/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2622"
 category_encoding_9/Assert/Const?
(category_encoding_9/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2622*
(category_encoding_9/Assert/Assert/data_0?
!category_encoding_9/Assert/AssertAssert"category_encoding_9/LogicalAnd:z:01category_encoding_9/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2#
!category_encoding_9/Assert/Assert?
"category_encoding_9/bincount/ShapeShapeAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_9/bincount/Shape?
"category_encoding_9/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_9/bincount/Const?
!category_encoding_9/bincount/ProdProd+category_encoding_9/bincount/Shape:output:0+category_encoding_9/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_9/bincount/Prod?
&category_encoding_9/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_9/bincount/Greater/y?
$category_encoding_9/bincount/GreaterGreater*category_encoding_9/bincount/Prod:output:0/category_encoding_9/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_9/bincount/Greater?
!category_encoding_9/bincount/CastCast(category_encoding_9/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_9/bincount/Cast?
$category_encoding_9/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_9/bincount/Const_1?
 category_encoding_9/bincount/MaxMaxAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_9/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_9/bincount/Max?
"category_encoding_9/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_9/bincount/add/y?
 category_encoding_9/bincount/addAddV2)category_encoding_9/bincount/Max:output:0+category_encoding_9/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_9/bincount/add?
 category_encoding_9/bincount/mulMul%category_encoding_9/bincount/Cast:y:0$category_encoding_9/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_9/bincount/mul?
&category_encoding_9/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_9/bincount/minlength?
$category_encoding_9/bincount/MaximumMaximum/category_encoding_9/bincount/minlength:output:0$category_encoding_9/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_9/bincount/Maximum?
&category_encoding_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_9/bincount/maxlength?
$category_encoding_9/bincount/MinimumMinimum/category_encoding_9/bincount/maxlength:output:0(category_encoding_9/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_9/bincount/Minimum?
$category_encoding_9/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_9/bincount/Const_2?
*category_encoding_9/bincount/DenseBincountDenseBincountAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_9/bincount/Minimum:z:0-category_encoding_9/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2,
*category_encoding_9/bincount/DenseBincount?
concatenate_19/PartitionedCallPartitionedCallnormalization_9/truediv:z:03category_encoding_9/bincount/DenseBincount:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_19_layer_call_and_return_conditional_losses_519652 
concatenate_19/PartitionedCall?
IdentityIdentity'concatenate_19/PartitionedCall:output:0"^category_encoding_9/Assert/Assert'^normalization_9/Reshape/ReadVariableOp)^normalization_9/Reshape_1/ReadVariableOp9^string_lookup_9/None_lookup_table_find/LookupTableFindV2*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 2F
!category_encoding_9/Assert/Assert!category_encoding_9/Assert/Assert2P
&normalization_9/Reshape/ReadVariableOp&normalization_9/Reshape/ReadVariableOp2T
(normalization_9/Reshape_1/ReadVariableOp(normalization_9/Reshape_1/ReadVariableOp2t
8string_lookup_9/None_lookup_table_find/LookupTableFindV28string_lookup_9/None_lookup_table_find/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
C__inference_model_19_layer_call_and_return_conditional_losses_52647

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
model_18_52620
model_18_52622	
model_18_52624:
model_18_52626:&
sequential_9_52629:
??!
sequential_9_52631:	?%
sequential_9_52633:	?@ 
sequential_9_52635:@$
sequential_9_52637:@  
sequential_9_52639: $
sequential_9_52641:  
sequential_9_52643:
identity?? model_18/StatefulPartitionedCall?$sequential_9/StatefulPartitionedCall?
 model_18/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6model_18_52620model_18_52622model_18_52624model_18_52626*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_18_layer_call_and_return_conditional_losses_520822"
 model_18/StatefulPartitionedCall?
$sequential_9/StatefulPartitionedCallStatefulPartitionedCall)model_18/StatefulPartitionedCall:output:0sequential_9_52629sequential_9_52631sequential_9_52633sequential_9_52635sequential_9_52637sequential_9_52639sequential_9_52641sequential_9_52643*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_524072&
$sequential_9/StatefulPartitionedCall?
IdentityIdentity-sequential_9/StatefulPartitionedCall:output:0!^model_18/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2D
 model_18/StatefulPartitionedCall model_18/StatefulPartitionedCall2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
(__inference_model_18_layer_call_fn_53222
inputs_area
inputs_garden_area
inputs_month_sold
inputs_rooms
inputs_square_meter
inputs_year_of_construction
inputs_year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_areainputs_garden_areainputs_month_soldinputs_roomsinputs_square_meterinputs_year_of_constructioninputs_year_soldunknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_18_layer_call_and_return_conditional_losses_520822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_nameinputs/area:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/garden_area:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/month_sold:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/rooms:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/square_meter:d`
'
_output_shapes
:?????????
5
_user_specified_nameinputs/year_of_construction:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/year_sold:

_output_shapes
: 
?
?
(__inference_model_19_layer_call_fn_52568
area
garden_area

month_sold	
rooms
square_meter
year_of_construction
	year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallareagarden_area
month_soldroomssquare_meteryear_of_construction	year_soldunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_19_layer_call_and_return_conditional_losses_525412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
'
_output_shapes
:?????????

_user_specified_namearea:TP
'
_output_shapes
:?????????
%
_user_specified_namegarden_area:SO
'
_output_shapes
:?????????
$
_user_specified_name
month_sold:NJ
'
_output_shapes
:?????????

_user_specified_namerooms:UQ
'
_output_shapes
:?????????
&
_user_specified_namesquare_meter:]Y
'
_output_shapes
:?????????
.
_user_specified_nameyear_of_construction:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:

_output_shapes
: 
?
?
(__inference_model_19_layer_call_fn_53029
inputs_area
inputs_garden_area
inputs_month_sold
inputs_rooms
inputs_square_meter
inputs_year_of_construction
inputs_year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_areainputs_garden_areainputs_month_soldinputs_roomsinputs_square_meterinputs_year_of_constructioninputs_year_soldunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_19_layer_call_and_return_conditional_losses_525412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_nameinputs/area:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/garden_area:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/month_sold:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/rooms:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/square_meter:d`
'
_output_shapes
:?????????
5
_user_specified_nameinputs/year_of_construction:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/year_sold:

_output_shapes
: 
?
?
(__inference_dense_38_layer_call_fn_53460

inputs
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_38_layer_call_and_return_conditional_losses_522782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_model_18_layer_call_fn_52112
area
garden_area

month_sold	
rooms
square_meter
year_of_construction
	year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallareagarden_area
month_soldroomssquare_meteryear_of_construction	year_soldunknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_18_layer_call_and_return_conditional_losses_520822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
'
_output_shapes
:?????????

_user_specified_namearea:TP
'
_output_shapes
:?????????
%
_user_specified_namegarden_area:SO
'
_output_shapes
:?????????
$
_user_specified_name
month_sold:NJ
'
_output_shapes
:?????????

_user_specified_namerooms:UQ
'
_output_shapes
:?????????
&
_user_specified_namesquare_meter:]Y
'
_output_shapes
:?????????
.
_user_specified_nameyear_of_construction:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:

_output_shapes
: 
?	
?
,__inference_sequential_9_layer_call_fn_52447
dense_36_input
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_36_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_524072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:??????????
(
_user_specified_namedense_36_input
??
?
C__inference_model_19_layer_call_and_return_conditional_losses_52994
inputs_area
inputs_garden_area
inputs_month_sold
inputs_rooms
inputs_square_meter
inputs_year_of_construction
inputs_year_soldR
Nmodel_18_string_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleS
Omodel_18_string_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value	F
8model_18_normalization_9_reshape_readvariableop_resource:H
:model_18_normalization_9_reshape_1_readvariableop_resource:H
4sequential_9_dense_36_matmul_readvariableop_resource:
??D
5sequential_9_dense_36_biasadd_readvariableop_resource:	?G
4sequential_9_dense_37_matmul_readvariableop_resource:	?@C
5sequential_9_dense_37_biasadd_readvariableop_resource:@F
4sequential_9_dense_38_matmul_readvariableop_resource:@ C
5sequential_9_dense_38_biasadd_readvariableop_resource: F
4sequential_9_dense_39_matmul_readvariableop_resource: C
5sequential_9_dense_39_biasadd_readvariableop_resource:
identity??*model_18/category_encoding_9/Assert/Assert?/model_18/normalization_9/Reshape/ReadVariableOp?1model_18/normalization_9/Reshape_1/ReadVariableOp?Amodel_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2?,sequential_9/dense_36/BiasAdd/ReadVariableOp?+sequential_9/dense_36/MatMul/ReadVariableOp?,sequential_9/dense_37/BiasAdd/ReadVariableOp?+sequential_9/dense_37/MatMul/ReadVariableOp?,sequential_9/dense_38/BiasAdd/ReadVariableOp?+sequential_9/dense_38/MatMul/ReadVariableOp?,sequential_9/dense_39/BiasAdd/ReadVariableOp?+sequential_9/dense_39/MatMul/ReadVariableOp?
Amodel_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Nmodel_18_string_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleinputs_areaOmodel_18_string_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2C
Amodel_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2?
#model_18/concatenate_18/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2%
#model_18/concatenate_18/concat/axis?
model_18/concatenate_18/concatConcatV2inputs_roomsinputs_square_meterinputs_garden_areainputs_year_of_constructioninputs_year_soldinputs_month_sold,model_18/concatenate_18/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2 
model_18/concatenate_18/concat?
/model_18/normalization_9/Reshape/ReadVariableOpReadVariableOp8model_18_normalization_9_reshape_readvariableop_resource*
_output_shapes
:*
dtype021
/model_18/normalization_9/Reshape/ReadVariableOp?
&model_18/normalization_9/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2(
&model_18/normalization_9/Reshape/shape?
 model_18/normalization_9/ReshapeReshape7model_18/normalization_9/Reshape/ReadVariableOp:value:0/model_18/normalization_9/Reshape/shape:output:0*
T0*
_output_shapes

:2"
 model_18/normalization_9/Reshape?
1model_18/normalization_9/Reshape_1/ReadVariableOpReadVariableOp:model_18_normalization_9_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype023
1model_18/normalization_9/Reshape_1/ReadVariableOp?
(model_18/normalization_9/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2*
(model_18/normalization_9/Reshape_1/shape?
"model_18/normalization_9/Reshape_1Reshape9model_18/normalization_9/Reshape_1/ReadVariableOp:value:01model_18/normalization_9/Reshape_1/shape:output:0*
T0*
_output_shapes

:2$
"model_18/normalization_9/Reshape_1?
model_18/normalization_9/subSub'model_18/concatenate_18/concat:output:0)model_18/normalization_9/Reshape:output:0*
T0*'
_output_shapes
:?????????2
model_18/normalization_9/sub?
model_18/normalization_9/SqrtSqrt+model_18/normalization_9/Reshape_1:output:0*
T0*
_output_shapes

:2
model_18/normalization_9/Sqrt?
"model_18/normalization_9/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32$
"model_18/normalization_9/Maximum/y?
 model_18/normalization_9/MaximumMaximum!model_18/normalization_9/Sqrt:y:0+model_18/normalization_9/Maximum/y:output:0*
T0*
_output_shapes

:2"
 model_18/normalization_9/Maximum?
 model_18/normalization_9/truedivRealDiv model_18/normalization_9/sub:z:0$model_18/normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????2"
 model_18/normalization_9/truediv?
"model_18/category_encoding_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model_18/category_encoding_9/Const?
 model_18/category_encoding_9/MaxMaxJmodel_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0+model_18/category_encoding_9/Const:output:0*
T0	*
_output_shapes
: 2"
 model_18/category_encoding_9/Max?
$model_18/category_encoding_9/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$model_18/category_encoding_9/Const_1?
 model_18/category_encoding_9/MinMinJmodel_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0-model_18/category_encoding_9/Const_1:output:0*
T0	*
_output_shapes
: 2"
 model_18/category_encoding_9/Min?
#model_18/category_encoding_9/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2%
#model_18/category_encoding_9/Cast/x?
!model_18/category_encoding_9/CastCast,model_18/category_encoding_9/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2#
!model_18/category_encoding_9/Cast?
$model_18/category_encoding_9/GreaterGreater%model_18/category_encoding_9/Cast:y:0)model_18/category_encoding_9/Max:output:0*
T0	*
_output_shapes
: 2&
$model_18/category_encoding_9/Greater?
%model_18/category_encoding_9/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2'
%model_18/category_encoding_9/Cast_1/x?
#model_18/category_encoding_9/Cast_1Cast.model_18/category_encoding_9/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2%
#model_18/category_encoding_9/Cast_1?
)model_18/category_encoding_9/GreaterEqualGreaterEqual)model_18/category_encoding_9/Min:output:0'model_18/category_encoding_9/Cast_1:y:0*
T0	*
_output_shapes
: 2+
)model_18/category_encoding_9/GreaterEqual?
'model_18/category_encoding_9/LogicalAnd
LogicalAnd(model_18/category_encoding_9/Greater:z:0-model_18/category_encoding_9/GreaterEqual:z:0*
_output_shapes
: 2)
'model_18/category_encoding_9/LogicalAnd?
)model_18/category_encoding_9/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2622+
)model_18/category_encoding_9/Assert/Const?
1model_18/category_encoding_9/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=26223
1model_18/category_encoding_9/Assert/Assert/data_0?
*model_18/category_encoding_9/Assert/AssertAssert+model_18/category_encoding_9/LogicalAnd:z:0:model_18/category_encoding_9/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2,
*model_18/category_encoding_9/Assert/Assert?
+model_18/category_encoding_9/bincount/ShapeShapeJmodel_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2-
+model_18/category_encoding_9/bincount/Shape?
+model_18/category_encoding_9/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+model_18/category_encoding_9/bincount/Const?
*model_18/category_encoding_9/bincount/ProdProd4model_18/category_encoding_9/bincount/Shape:output:04model_18/category_encoding_9/bincount/Const:output:0*
T0*
_output_shapes
: 2,
*model_18/category_encoding_9/bincount/Prod?
/model_18/category_encoding_9/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 21
/model_18/category_encoding_9/bincount/Greater/y?
-model_18/category_encoding_9/bincount/GreaterGreater3model_18/category_encoding_9/bincount/Prod:output:08model_18/category_encoding_9/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2/
-model_18/category_encoding_9/bincount/Greater?
*model_18/category_encoding_9/bincount/CastCast1model_18/category_encoding_9/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2,
*model_18/category_encoding_9/bincount/Cast?
-model_18/category_encoding_9/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-model_18/category_encoding_9/bincount/Const_1?
)model_18/category_encoding_9/bincount/MaxMaxJmodel_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2:values:06model_18/category_encoding_9/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2+
)model_18/category_encoding_9/bincount/Max?
+model_18/category_encoding_9/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2-
+model_18/category_encoding_9/bincount/add/y?
)model_18/category_encoding_9/bincount/addAddV22model_18/category_encoding_9/bincount/Max:output:04model_18/category_encoding_9/bincount/add/y:output:0*
T0	*
_output_shapes
: 2+
)model_18/category_encoding_9/bincount/add?
)model_18/category_encoding_9/bincount/mulMul.model_18/category_encoding_9/bincount/Cast:y:0-model_18/category_encoding_9/bincount/add:z:0*
T0	*
_output_shapes
: 2+
)model_18/category_encoding_9/bincount/mul?
/model_18/category_encoding_9/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?21
/model_18/category_encoding_9/bincount/minlength?
-model_18/category_encoding_9/bincount/MaximumMaximum8model_18/category_encoding_9/bincount/minlength:output:0-model_18/category_encoding_9/bincount/mul:z:0*
T0	*
_output_shapes
: 2/
-model_18/category_encoding_9/bincount/Maximum?
/model_18/category_encoding_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?21
/model_18/category_encoding_9/bincount/maxlength?
-model_18/category_encoding_9/bincount/MinimumMinimum8model_18/category_encoding_9/bincount/maxlength:output:01model_18/category_encoding_9/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2/
-model_18/category_encoding_9/bincount/Minimum?
-model_18/category_encoding_9/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2/
-model_18/category_encoding_9/bincount/Const_2?
3model_18/category_encoding_9/bincount/DenseBincountDenseBincountJmodel_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2:values:01model_18/category_encoding_9/bincount/Minimum:z:06model_18/category_encoding_9/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(25
3model_18/category_encoding_9/bincount/DenseBincount?
#model_18/concatenate_19/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2%
#model_18/concatenate_19/concat/axis?
model_18/concatenate_19/concatConcatV2$model_18/normalization_9/truediv:z:0<model_18/category_encoding_9/bincount/DenseBincount:output:0,model_18/concatenate_19/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2 
model_18/concatenate_19/concat?
+sequential_9/dense_36/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_36_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+sequential_9/dense_36/MatMul/ReadVariableOp?
sequential_9/dense_36/MatMulMatMul'model_18/concatenate_19/concat:output:03sequential_9/dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_36/MatMul?
,sequential_9/dense_36/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential_9/dense_36/BiasAdd/ReadVariableOp?
sequential_9/dense_36/BiasAddBiasAdd&sequential_9/dense_36/MatMul:product:04sequential_9/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_36/BiasAdd?
sequential_9/dense_36/TanhTanh&sequential_9/dense_36/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_36/Tanh?
+sequential_9/dense_37/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_37_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02-
+sequential_9/dense_37/MatMul/ReadVariableOp?
sequential_9/dense_37/MatMulMatMulsequential_9/dense_36/Tanh:y:03sequential_9/dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential_9/dense_37/MatMul?
,sequential_9/dense_37/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_9/dense_37/BiasAdd/ReadVariableOp?
sequential_9/dense_37/BiasAddBiasAdd&sequential_9/dense_37/MatMul:product:04sequential_9/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential_9/dense_37/BiasAdd?
+sequential_9/dense_38/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_38_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02-
+sequential_9/dense_38/MatMul/ReadVariableOp?
sequential_9/dense_38/MatMulMatMul&sequential_9/dense_37/BiasAdd:output:03sequential_9/dense_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_9/dense_38/MatMul?
,sequential_9/dense_38/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_38_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_9/dense_38/BiasAdd/ReadVariableOp?
sequential_9/dense_38/BiasAddBiasAdd&sequential_9/dense_38/MatMul:product:04sequential_9/dense_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_9/dense_38/BiasAdd?
+sequential_9/dense_39/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_39_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+sequential_9/dense_39/MatMul/ReadVariableOp?
sequential_9/dense_39/MatMulMatMul&sequential_9/dense_38/BiasAdd:output:03sequential_9/dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_9/dense_39/MatMul?
,sequential_9/dense_39/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_9/dense_39/BiasAdd/ReadVariableOp?
sequential_9/dense_39/BiasAddBiasAdd&sequential_9/dense_39/MatMul:product:04sequential_9/dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_9/dense_39/BiasAdd?
IdentityIdentity&sequential_9/dense_39/BiasAdd:output:0+^model_18/category_encoding_9/Assert/Assert0^model_18/normalization_9/Reshape/ReadVariableOp2^model_18/normalization_9/Reshape_1/ReadVariableOpB^model_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2-^sequential_9/dense_36/BiasAdd/ReadVariableOp,^sequential_9/dense_36/MatMul/ReadVariableOp-^sequential_9/dense_37/BiasAdd/ReadVariableOp,^sequential_9/dense_37/MatMul/ReadVariableOp-^sequential_9/dense_38/BiasAdd/ReadVariableOp,^sequential_9/dense_38/MatMul/ReadVariableOp-^sequential_9/dense_39/BiasAdd/ReadVariableOp,^sequential_9/dense_39/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2X
*model_18/category_encoding_9/Assert/Assert*model_18/category_encoding_9/Assert/Assert2b
/model_18/normalization_9/Reshape/ReadVariableOp/model_18/normalization_9/Reshape/ReadVariableOp2f
1model_18/normalization_9/Reshape_1/ReadVariableOp1model_18/normalization_9/Reshape_1/ReadVariableOp2?
Amodel_18/string_lookup_9/None_lookup_table_find/LookupTableFindV2Amodel_18/string_lookup_9/None_lookup_table_find/LookupTableFindV22\
,sequential_9/dense_36/BiasAdd/ReadVariableOp,sequential_9/dense_36/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_36/MatMul/ReadVariableOp+sequential_9/dense_36/MatMul/ReadVariableOp2\
,sequential_9/dense_37/BiasAdd/ReadVariableOp,sequential_9/dense_37/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_37/MatMul/ReadVariableOp+sequential_9/dense_37/MatMul/ReadVariableOp2\
,sequential_9/dense_38/BiasAdd/ReadVariableOp,sequential_9/dense_38/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_38/MatMul/ReadVariableOp+sequential_9/dense_38/MatMul/ReadVariableOp2\
,sequential_9/dense_39/BiasAdd/ReadVariableOp,sequential_9/dense_39/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_39/MatMul/ReadVariableOp+sequential_9/dense_39/MatMul/ReadVariableOp:T P
'
_output_shapes
:?????????
%
_user_specified_nameinputs/area:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/garden_area:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/month_sold:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/rooms:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/square_meter:d`
'
_output_shapes
:?????????
5
_user_specified_nameinputs/year_of_construction:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/year_sold:

_output_shapes
: 
?T
?
C__inference_model_18_layer_call_and_return_conditional_losses_52082

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6I
Estring_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value	=
/normalization_9_reshape_readvariableop_resource:?
1normalization_9_reshape_1_readvariableop_resource:
identity??!category_encoding_9/Assert/Assert?&normalization_9/Reshape/ReadVariableOp?(normalization_9/Reshape_1/ReadVariableOp?8string_lookup_9/None_lookup_table_find/LookupTableFindV2?
8string_lookup_9/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleinputsFstring_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_9/None_lookup_table_find/LookupTableFindV2?
concatenate_18/PartitionedCallPartitionedCallinputs_3inputs_4inputs_1inputs_5inputs_6inputs_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_18_layer_call_and_return_conditional_losses_519132 
concatenate_18/PartitionedCall?
&normalization_9/Reshape/ReadVariableOpReadVariableOp/normalization_9_reshape_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization_9/Reshape/ReadVariableOp?
normalization_9/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_9/Reshape/shape?
normalization_9/ReshapeReshape.normalization_9/Reshape/ReadVariableOp:value:0&normalization_9/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_9/Reshape?
(normalization_9/Reshape_1/ReadVariableOpReadVariableOp1normalization_9_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02*
(normalization_9/Reshape_1/ReadVariableOp?
normalization_9/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_9/Reshape_1/shape?
normalization_9/Reshape_1Reshape0normalization_9/Reshape_1/ReadVariableOp:value:0(normalization_9/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_9/Reshape_1?
normalization_9/subSub'concatenate_18/PartitionedCall:output:0 normalization_9/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization_9/sub?
normalization_9/SqrtSqrt"normalization_9/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_9/Sqrt{
normalization_9/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_9/Maximum/y?
normalization_9/MaximumMaximumnormalization_9/Sqrt:y:0"normalization_9/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_9/Maximum?
normalization_9/truedivRealDivnormalization_9/sub:z:0normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_9/truediv?
category_encoding_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_9/Const?
category_encoding_9/MaxMaxAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0"category_encoding_9/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_9/Max?
category_encoding_9/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_9/Const_1?
category_encoding_9/MinMinAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0$category_encoding_9/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_9/Min{
category_encoding_9/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding_9/Cast/x?
category_encoding_9/CastCast#category_encoding_9/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_9/Cast?
category_encoding_9/GreaterGreatercategory_encoding_9/Cast:y:0 category_encoding_9/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_9/Greater~
category_encoding_9/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_9/Cast_1/x?
category_encoding_9/Cast_1Cast%category_encoding_9/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_9/Cast_1?
 category_encoding_9/GreaterEqualGreaterEqual category_encoding_9/Min:output:0category_encoding_9/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_9/GreaterEqual?
category_encoding_9/LogicalAnd
LogicalAndcategory_encoding_9/Greater:z:0$category_encoding_9/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_9/LogicalAnd?
 category_encoding_9/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2622"
 category_encoding_9/Assert/Const?
(category_encoding_9/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2622*
(category_encoding_9/Assert/Assert/data_0?
!category_encoding_9/Assert/AssertAssert"category_encoding_9/LogicalAnd:z:01category_encoding_9/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2#
!category_encoding_9/Assert/Assert?
"category_encoding_9/bincount/ShapeShapeAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_9/bincount/Shape?
"category_encoding_9/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_9/bincount/Const?
!category_encoding_9/bincount/ProdProd+category_encoding_9/bincount/Shape:output:0+category_encoding_9/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_9/bincount/Prod?
&category_encoding_9/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_9/bincount/Greater/y?
$category_encoding_9/bincount/GreaterGreater*category_encoding_9/bincount/Prod:output:0/category_encoding_9/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_9/bincount/Greater?
!category_encoding_9/bincount/CastCast(category_encoding_9/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_9/bincount/Cast?
$category_encoding_9/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_9/bincount/Const_1?
 category_encoding_9/bincount/MaxMaxAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_9/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_9/bincount/Max?
"category_encoding_9/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_9/bincount/add/y?
 category_encoding_9/bincount/addAddV2)category_encoding_9/bincount/Max:output:0+category_encoding_9/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_9/bincount/add?
 category_encoding_9/bincount/mulMul%category_encoding_9/bincount/Cast:y:0$category_encoding_9/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_9/bincount/mul?
&category_encoding_9/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_9/bincount/minlength?
$category_encoding_9/bincount/MaximumMaximum/category_encoding_9/bincount/minlength:output:0$category_encoding_9/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_9/bincount/Maximum?
&category_encoding_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_9/bincount/maxlength?
$category_encoding_9/bincount/MinimumMinimum/category_encoding_9/bincount/maxlength:output:0(category_encoding_9/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_9/bincount/Minimum?
$category_encoding_9/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_9/bincount/Const_2?
*category_encoding_9/bincount/DenseBincountDenseBincountAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_9/bincount/Minimum:z:0-category_encoding_9/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2,
*category_encoding_9/bincount/DenseBincount?
concatenate_19/PartitionedCallPartitionedCallnormalization_9/truediv:z:03category_encoding_9/bincount/DenseBincount:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_19_layer_call_and_return_conditional_losses_519652 
concatenate_19/PartitionedCall?
IdentityIdentity'concatenate_19/PartitionedCall:output:0"^category_encoding_9/Assert/Assert'^normalization_9/Reshape/ReadVariableOp)^normalization_9/Reshape_1/ReadVariableOp9^string_lookup_9/None_lookup_table_find/LookupTableFindV2*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 2F
!category_encoding_9/Assert/Assert!category_encoding_9/Assert/Assert2P
&normalization_9/Reshape/ReadVariableOp&normalization_9/Reshape/ReadVariableOp2T
(normalization_9/Reshape_1/ReadVariableOp(normalization_9/Reshape_1/ReadVariableOp2t
8string_lookup_9/None_lookup_table_find/LookupTableFindV28string_lookup_9/None_lookup_table_find/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_52471
dense_36_input"
dense_36_52450:
??
dense_36_52452:	?!
dense_37_52455:	?@
dense_37_52457:@ 
dense_38_52460:@ 
dense_38_52462:  
dense_39_52465: 
dense_39_52467:
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall? dense_38/StatefulPartitionedCall? dense_39/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCalldense_36_inputdense_36_52450dense_36_52452*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_36_layer_call_and_return_conditional_losses_522462"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_52455dense_37_52457*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_37_layer_call_and_return_conditional_losses_522622"
 dense_37/StatefulPartitionedCall?
 dense_38/StatefulPartitionedCallStatefulPartitionedCall)dense_37/StatefulPartitionedCall:output:0dense_38_52460dense_38_52462*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_38_layer_call_and_return_conditional_losses_522782"
 dense_38/StatefulPartitionedCall?
 dense_39/StatefulPartitionedCallStatefulPartitionedCall)dense_38/StatefulPartitionedCall:output:0dense_39_52465dense_39_52467*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_39_layer_call_and_return_conditional_losses_522942"
 dense_39/StatefulPartitionedCall?
IdentityIdentity)dense_39/StatefulPartitionedCall:output:0!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall:X T
(
_output_shapes
:??????????
(
_user_specified_namedense_36_input
?	
?
I__inference_concatenate_18_layer_call_and_return_conditional_losses_53333
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapest
r:?????????:?????????:?????????:?????????:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5
?U
?
C__inference_model_18_layer_call_and_return_conditional_losses_52228
area
garden_area

month_sold	
rooms
square_meter
year_of_construction
	year_soldI
Estring_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value	=
/normalization_9_reshape_readvariableop_resource:?
1normalization_9_reshape_1_readvariableop_resource:
identity??!category_encoding_9/Assert/Assert?&normalization_9/Reshape/ReadVariableOp?(normalization_9/Reshape_1/ReadVariableOp?8string_lookup_9/None_lookup_table_find/LookupTableFindV2?
8string_lookup_9/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleareaFstring_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_9/None_lookup_table_find/LookupTableFindV2?
concatenate_18/PartitionedCallPartitionedCallroomssquare_metergarden_areayear_of_construction	year_sold
month_sold*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_18_layer_call_and_return_conditional_losses_519132 
concatenate_18/PartitionedCall?
&normalization_9/Reshape/ReadVariableOpReadVariableOp/normalization_9_reshape_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization_9/Reshape/ReadVariableOp?
normalization_9/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_9/Reshape/shape?
normalization_9/ReshapeReshape.normalization_9/Reshape/ReadVariableOp:value:0&normalization_9/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_9/Reshape?
(normalization_9/Reshape_1/ReadVariableOpReadVariableOp1normalization_9_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02*
(normalization_9/Reshape_1/ReadVariableOp?
normalization_9/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_9/Reshape_1/shape?
normalization_9/Reshape_1Reshape0normalization_9/Reshape_1/ReadVariableOp:value:0(normalization_9/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_9/Reshape_1?
normalization_9/subSub'concatenate_18/PartitionedCall:output:0 normalization_9/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization_9/sub?
normalization_9/SqrtSqrt"normalization_9/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_9/Sqrt{
normalization_9/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_9/Maximum/y?
normalization_9/MaximumMaximumnormalization_9/Sqrt:y:0"normalization_9/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_9/Maximum?
normalization_9/truedivRealDivnormalization_9/sub:z:0normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_9/truediv?
category_encoding_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_9/Const?
category_encoding_9/MaxMaxAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0"category_encoding_9/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_9/Max?
category_encoding_9/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_9/Const_1?
category_encoding_9/MinMinAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0$category_encoding_9/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_9/Min{
category_encoding_9/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding_9/Cast/x?
category_encoding_9/CastCast#category_encoding_9/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_9/Cast?
category_encoding_9/GreaterGreatercategory_encoding_9/Cast:y:0 category_encoding_9/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_9/Greater~
category_encoding_9/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_9/Cast_1/x?
category_encoding_9/Cast_1Cast%category_encoding_9/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_9/Cast_1?
 category_encoding_9/GreaterEqualGreaterEqual category_encoding_9/Min:output:0category_encoding_9/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_9/GreaterEqual?
category_encoding_9/LogicalAnd
LogicalAndcategory_encoding_9/Greater:z:0$category_encoding_9/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_9/LogicalAnd?
 category_encoding_9/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2622"
 category_encoding_9/Assert/Const?
(category_encoding_9/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2622*
(category_encoding_9/Assert/Assert/data_0?
!category_encoding_9/Assert/AssertAssert"category_encoding_9/LogicalAnd:z:01category_encoding_9/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2#
!category_encoding_9/Assert/Assert?
"category_encoding_9/bincount/ShapeShapeAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_9/bincount/Shape?
"category_encoding_9/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_9/bincount/Const?
!category_encoding_9/bincount/ProdProd+category_encoding_9/bincount/Shape:output:0+category_encoding_9/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_9/bincount/Prod?
&category_encoding_9/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_9/bincount/Greater/y?
$category_encoding_9/bincount/GreaterGreater*category_encoding_9/bincount/Prod:output:0/category_encoding_9/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_9/bincount/Greater?
!category_encoding_9/bincount/CastCast(category_encoding_9/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_9/bincount/Cast?
$category_encoding_9/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_9/bincount/Const_1?
 category_encoding_9/bincount/MaxMaxAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_9/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_9/bincount/Max?
"category_encoding_9/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_9/bincount/add/y?
 category_encoding_9/bincount/addAddV2)category_encoding_9/bincount/Max:output:0+category_encoding_9/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_9/bincount/add?
 category_encoding_9/bincount/mulMul%category_encoding_9/bincount/Cast:y:0$category_encoding_9/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_9/bincount/mul?
&category_encoding_9/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_9/bincount/minlength?
$category_encoding_9/bincount/MaximumMaximum/category_encoding_9/bincount/minlength:output:0$category_encoding_9/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_9/bincount/Maximum?
&category_encoding_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_9/bincount/maxlength?
$category_encoding_9/bincount/MinimumMinimum/category_encoding_9/bincount/maxlength:output:0(category_encoding_9/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_9/bincount/Minimum?
$category_encoding_9/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_9/bincount/Const_2?
*category_encoding_9/bincount/DenseBincountDenseBincountAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_9/bincount/Minimum:z:0-category_encoding_9/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2,
*category_encoding_9/bincount/DenseBincount?
concatenate_19/PartitionedCallPartitionedCallnormalization_9/truediv:z:03category_encoding_9/bincount/DenseBincount:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_19_layer_call_and_return_conditional_losses_519652 
concatenate_19/PartitionedCall?
IdentityIdentity'concatenate_19/PartitionedCall:output:0"^category_encoding_9/Assert/Assert'^normalization_9/Reshape/ReadVariableOp)^normalization_9/Reshape_1/ReadVariableOp9^string_lookup_9/None_lookup_table_find/LookupTableFindV2*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 2F
!category_encoding_9/Assert/Assert!category_encoding_9/Assert/Assert2P
&normalization_9/Reshape/ReadVariableOp&normalization_9/Reshape/ReadVariableOp2T
(normalization_9/Reshape_1/ReadVariableOp(normalization_9/Reshape_1/ReadVariableOp2t
8string_lookup_9/None_lookup_table_find/LookupTableFindV28string_lookup_9/None_lookup_table_find/LookupTableFindV2:M I
'
_output_shapes
:?????????

_user_specified_namearea:TP
'
_output_shapes
:?????????
%
_user_specified_namegarden_area:SO
'
_output_shapes
:?????????
$
_user_specified_name
month_sold:NJ
'
_output_shapes
:?????????

_user_specified_namerooms:UQ
'
_output_shapes
:?????????
&
_user_specified_namesquare_meter:]Y
'
_output_shapes
:?????????
.
_user_specified_nameyear_of_construction:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:

_output_shapes
: 
?U
?
C__inference_model_18_layer_call_and_return_conditional_losses_52170
area
garden_area

month_sold	
rooms
square_meter
year_of_construction
	year_soldI
Estring_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value	=
/normalization_9_reshape_readvariableop_resource:?
1normalization_9_reshape_1_readvariableop_resource:
identity??!category_encoding_9/Assert/Assert?&normalization_9/Reshape/ReadVariableOp?(normalization_9/Reshape_1/ReadVariableOp?8string_lookup_9/None_lookup_table_find/LookupTableFindV2?
8string_lookup_9/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleareaFstring_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_9/None_lookup_table_find/LookupTableFindV2?
concatenate_18/PartitionedCallPartitionedCallroomssquare_metergarden_areayear_of_construction	year_sold
month_sold*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_18_layer_call_and_return_conditional_losses_519132 
concatenate_18/PartitionedCall?
&normalization_9/Reshape/ReadVariableOpReadVariableOp/normalization_9_reshape_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization_9/Reshape/ReadVariableOp?
normalization_9/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_9/Reshape/shape?
normalization_9/ReshapeReshape.normalization_9/Reshape/ReadVariableOp:value:0&normalization_9/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_9/Reshape?
(normalization_9/Reshape_1/ReadVariableOpReadVariableOp1normalization_9_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02*
(normalization_9/Reshape_1/ReadVariableOp?
normalization_9/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_9/Reshape_1/shape?
normalization_9/Reshape_1Reshape0normalization_9/Reshape_1/ReadVariableOp:value:0(normalization_9/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_9/Reshape_1?
normalization_9/subSub'concatenate_18/PartitionedCall:output:0 normalization_9/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization_9/sub?
normalization_9/SqrtSqrt"normalization_9/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_9/Sqrt{
normalization_9/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_9/Maximum/y?
normalization_9/MaximumMaximumnormalization_9/Sqrt:y:0"normalization_9/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_9/Maximum?
normalization_9/truedivRealDivnormalization_9/sub:z:0normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_9/truediv?
category_encoding_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_9/Const?
category_encoding_9/MaxMaxAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0"category_encoding_9/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_9/Max?
category_encoding_9/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_9/Const_1?
category_encoding_9/MinMinAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0$category_encoding_9/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_9/Min{
category_encoding_9/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding_9/Cast/x?
category_encoding_9/CastCast#category_encoding_9/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_9/Cast?
category_encoding_9/GreaterGreatercategory_encoding_9/Cast:y:0 category_encoding_9/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_9/Greater~
category_encoding_9/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_9/Cast_1/x?
category_encoding_9/Cast_1Cast%category_encoding_9/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_9/Cast_1?
 category_encoding_9/GreaterEqualGreaterEqual category_encoding_9/Min:output:0category_encoding_9/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_9/GreaterEqual?
category_encoding_9/LogicalAnd
LogicalAndcategory_encoding_9/Greater:z:0$category_encoding_9/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_9/LogicalAnd?
 category_encoding_9/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2622"
 category_encoding_9/Assert/Const?
(category_encoding_9/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2622*
(category_encoding_9/Assert/Assert/data_0?
!category_encoding_9/Assert/AssertAssert"category_encoding_9/LogicalAnd:z:01category_encoding_9/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2#
!category_encoding_9/Assert/Assert?
"category_encoding_9/bincount/ShapeShapeAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_9/bincount/Shape?
"category_encoding_9/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_9/bincount/Const?
!category_encoding_9/bincount/ProdProd+category_encoding_9/bincount/Shape:output:0+category_encoding_9/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_9/bincount/Prod?
&category_encoding_9/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_9/bincount/Greater/y?
$category_encoding_9/bincount/GreaterGreater*category_encoding_9/bincount/Prod:output:0/category_encoding_9/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_9/bincount/Greater?
!category_encoding_9/bincount/CastCast(category_encoding_9/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_9/bincount/Cast?
$category_encoding_9/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_9/bincount/Const_1?
 category_encoding_9/bincount/MaxMaxAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_9/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_9/bincount/Max?
"category_encoding_9/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_9/bincount/add/y?
 category_encoding_9/bincount/addAddV2)category_encoding_9/bincount/Max:output:0+category_encoding_9/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_9/bincount/add?
 category_encoding_9/bincount/mulMul%category_encoding_9/bincount/Cast:y:0$category_encoding_9/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_9/bincount/mul?
&category_encoding_9/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_9/bincount/minlength?
$category_encoding_9/bincount/MaximumMaximum/category_encoding_9/bincount/minlength:output:0$category_encoding_9/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_9/bincount/Maximum?
&category_encoding_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_9/bincount/maxlength?
$category_encoding_9/bincount/MinimumMinimum/category_encoding_9/bincount/maxlength:output:0(category_encoding_9/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_9/bincount/Minimum?
$category_encoding_9/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_9/bincount/Const_2?
*category_encoding_9/bincount/DenseBincountDenseBincountAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_9/bincount/Minimum:z:0-category_encoding_9/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2,
*category_encoding_9/bincount/DenseBincount?
concatenate_19/PartitionedCallPartitionedCallnormalization_9/truediv:z:03category_encoding_9/bincount/DenseBincount:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_19_layer_call_and_return_conditional_losses_519652 
concatenate_19/PartitionedCall?
IdentityIdentity'concatenate_19/PartitionedCall:output:0"^category_encoding_9/Assert/Assert'^normalization_9/Reshape/ReadVariableOp)^normalization_9/Reshape_1/ReadVariableOp9^string_lookup_9/None_lookup_table_find/LookupTableFindV2*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 2F
!category_encoding_9/Assert/Assert!category_encoding_9/Assert/Assert2P
&normalization_9/Reshape/ReadVariableOp&normalization_9/Reshape/ReadVariableOp2T
(normalization_9/Reshape_1/ReadVariableOp(normalization_9/Reshape_1/ReadVariableOp2t
8string_lookup_9/None_lookup_table_find/LookupTableFindV28string_lookup_9/None_lookup_table_find/LookupTableFindV2:M I
'
_output_shapes
:?????????

_user_specified_namearea:TP
'
_output_shapes
:?????????
%
_user_specified_namegarden_area:SO
'
_output_shapes
:?????????
$
_user_specified_name
month_sold:NJ
'
_output_shapes
:?????????

_user_specified_namerooms:UQ
'
_output_shapes
:?????????
&
_user_specified_namesquare_meter:]Y
'
_output_shapes
:?????????
.
_user_specified_nameyear_of_construction:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:

_output_shapes
: 
?%
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_53280

inputs;
'dense_36_matmul_readvariableop_resource:
??7
(dense_36_biasadd_readvariableop_resource:	?:
'dense_37_matmul_readvariableop_resource:	?@6
(dense_37_biasadd_readvariableop_resource:@9
'dense_38_matmul_readvariableop_resource:@ 6
(dense_38_biasadd_readvariableop_resource: 9
'dense_39_matmul_readvariableop_resource: 6
(dense_39_biasadd_readvariableop_resource:
identity??dense_36/BiasAdd/ReadVariableOp?dense_36/MatMul/ReadVariableOp?dense_37/BiasAdd/ReadVariableOp?dense_37/MatMul/ReadVariableOp?dense_38/BiasAdd/ReadVariableOp?dense_38/MatMul/ReadVariableOp?dense_39/BiasAdd/ReadVariableOp?dense_39/MatMul/ReadVariableOp?
dense_36/MatMul/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_36/MatMul/ReadVariableOp?
dense_36/MatMulMatMulinputs&dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_36/MatMul?
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_36/BiasAdd/ReadVariableOp?
dense_36/BiasAddBiasAdddense_36/MatMul:product:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_36/BiasAddt
dense_36/TanhTanhdense_36/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_36/Tanh?
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02 
dense_37/MatMul/ReadVariableOp?
dense_37/MatMulMatMuldense_36/Tanh:y:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_37/MatMul?
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_37/BiasAdd/ReadVariableOp?
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_37/BiasAdd?
dense_38/MatMul/ReadVariableOpReadVariableOp'dense_38_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02 
dense_38/MatMul/ReadVariableOp?
dense_38/MatMulMatMuldense_37/BiasAdd:output:0&dense_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_38/MatMul?
dense_38/BiasAdd/ReadVariableOpReadVariableOp(dense_38_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_38/BiasAdd/ReadVariableOp?
dense_38/BiasAddBiasAdddense_38/MatMul:product:0'dense_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_38/BiasAdd?
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_39/MatMul/ReadVariableOp?
dense_39/MatMulMatMuldense_38/BiasAdd:output:0&dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_39/MatMul?
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_39/BiasAdd/ReadVariableOp?
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_39/BiasAdd?
IdentityIdentitydense_39/BiasAdd:output:0 ^dense_36/BiasAdd/ReadVariableOp^dense_36/MatMul/ReadVariableOp ^dense_37/BiasAdd/ReadVariableOp^dense_37/MatMul/ReadVariableOp ^dense_38/BiasAdd/ReadVariableOp^dense_38/MatMul/ReadVariableOp ^dense_39/BiasAdd/ReadVariableOp^dense_39/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2B
dense_36/BiasAdd/ReadVariableOpdense_36/BiasAdd/ReadVariableOp2@
dense_36/MatMul/ReadVariableOpdense_36/MatMul/ReadVariableOp2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2@
dense_37/MatMul/ReadVariableOpdense_37/MatMul/ReadVariableOp2B
dense_38/BiasAdd/ReadVariableOpdense_38/BiasAdd/ReadVariableOp2@
dense_38/MatMul/ReadVariableOpdense_38/MatMul/ReadVariableOp2B
dense_39/BiasAdd/ReadVariableOpdense_39/BiasAdd/ReadVariableOp2@
dense_39/MatMul/ReadVariableOpdense_39/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_52495
dense_36_input"
dense_36_52474:
??
dense_36_52476:	?!
dense_37_52479:	?@
dense_37_52481:@ 
dense_38_52484:@ 
dense_38_52486:  
dense_39_52489: 
dense_39_52491:
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall? dense_38/StatefulPartitionedCall? dense_39/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCalldense_36_inputdense_36_52474dense_36_52476*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_36_layer_call_and_return_conditional_losses_522462"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_52479dense_37_52481*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_37_layer_call_and_return_conditional_losses_522622"
 dense_37/StatefulPartitionedCall?
 dense_38/StatefulPartitionedCallStatefulPartitionedCall)dense_37/StatefulPartitionedCall:output:0dense_38_52484dense_38_52486*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_38_layer_call_and_return_conditional_losses_522782"
 dense_38/StatefulPartitionedCall?
 dense_39/StatefulPartitionedCallStatefulPartitionedCall)dense_38/StatefulPartitionedCall:output:0dense_39_52489dense_39_52491*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_39_layer_call_and_return_conditional_losses_522942"
 dense_39/StatefulPartitionedCall?
IdentityIdentity)dense_39/StatefulPartitionedCall:output:0!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall:X T
(
_output_shapes
:??????????
(
_user_specified_namedense_36_input
?
?
C__inference_model_19_layer_call_and_return_conditional_losses_52781
area
garden_area

month_sold	
rooms
square_meter
year_of_construction
	year_sold
model_18_52754
model_18_52756	
model_18_52758:
model_18_52760:&
sequential_9_52763:
??!
sequential_9_52765:	?%
sequential_9_52767:	?@ 
sequential_9_52769:@$
sequential_9_52771:@  
sequential_9_52773: $
sequential_9_52775:  
sequential_9_52777:
identity?? model_18/StatefulPartitionedCall?$sequential_9/StatefulPartitionedCall?
 model_18/StatefulPartitionedCallStatefulPartitionedCallareagarden_area
month_soldroomssquare_meteryear_of_construction	year_soldmodel_18_52754model_18_52756model_18_52758model_18_52760*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_18_layer_call_and_return_conditional_losses_520822"
 model_18/StatefulPartitionedCall?
$sequential_9/StatefulPartitionedCallStatefulPartitionedCall)model_18/StatefulPartitionedCall:output:0sequential_9_52763sequential_9_52765sequential_9_52767sequential_9_52769sequential_9_52771sequential_9_52773sequential_9_52775sequential_9_52777*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_524072&
$sequential_9/StatefulPartitionedCall?
IdentityIdentity-sequential_9/StatefulPartitionedCall:output:0!^model_18/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2D
 model_18/StatefulPartitionedCall model_18/StatefulPartitionedCall2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:M I
'
_output_shapes
:?????????

_user_specified_namearea:TP
'
_output_shapes
:?????????
%
_user_specified_namegarden_area:SO
'
_output_shapes
:?????????
$
_user_specified_name
month_sold:NJ
'
_output_shapes
:?????????

_user_specified_namerooms:UQ
'
_output_shapes
:?????????
&
_user_specified_namesquare_meter:]Y
'
_output_shapes
:?????????
.
_user_specified_nameyear_of_construction:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:

_output_shapes
: 
?
?
(__inference_dense_37_layer_call_fn_53441

inputs
unknown:	?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_37_layer_call_and_return_conditional_losses_522622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
__inference_restore_fn_53521
restored_tensors_0
restored_tensors_1	N
Jstring_lookup_9_index_table_table_restore_lookuptableimportv2_table_handle
identity??=string_lookup_9_index_table_table_restore/LookupTableImportV2?
=string_lookup_9_index_table_table_restore/LookupTableImportV2LookupTableImportV2Jstring_lookup_9_index_table_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 2?
=string_lookup_9_index_table_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Const?
IdentityIdentityConst:output:0>^string_lookup_9_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2~
=string_lookup_9_index_table_table_restore/LookupTableImportV2=string_lookup_9_index_table_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?U
?
C__inference_model_18_layer_call_and_return_conditional_losses_53184
inputs_area
inputs_garden_area
inputs_month_sold
inputs_rooms
inputs_square_meter
inputs_year_of_construction
inputs_year_soldI
Estring_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value	=
/normalization_9_reshape_readvariableop_resource:?
1normalization_9_reshape_1_readvariableop_resource:
identity??!category_encoding_9/Assert/Assert?&normalization_9/Reshape/ReadVariableOp?(normalization_9/Reshape_1/ReadVariableOp?8string_lookup_9/None_lookup_table_find/LookupTableFindV2?
8string_lookup_9/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleinputs_areaFstring_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_9/None_lookup_table_find/LookupTableFindV2z
concatenate_18/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_18/concat/axis?
concatenate_18/concatConcatV2inputs_roomsinputs_square_meterinputs_garden_areainputs_year_of_constructioninputs_year_soldinputs_month_sold#concatenate_18/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate_18/concat?
&normalization_9/Reshape/ReadVariableOpReadVariableOp/normalization_9_reshape_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization_9/Reshape/ReadVariableOp?
normalization_9/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_9/Reshape/shape?
normalization_9/ReshapeReshape.normalization_9/Reshape/ReadVariableOp:value:0&normalization_9/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_9/Reshape?
(normalization_9/Reshape_1/ReadVariableOpReadVariableOp1normalization_9_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02*
(normalization_9/Reshape_1/ReadVariableOp?
normalization_9/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_9/Reshape_1/shape?
normalization_9/Reshape_1Reshape0normalization_9/Reshape_1/ReadVariableOp:value:0(normalization_9/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_9/Reshape_1?
normalization_9/subSubconcatenate_18/concat:output:0 normalization_9/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization_9/sub?
normalization_9/SqrtSqrt"normalization_9/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_9/Sqrt{
normalization_9/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_9/Maximum/y?
normalization_9/MaximumMaximumnormalization_9/Sqrt:y:0"normalization_9/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_9/Maximum?
normalization_9/truedivRealDivnormalization_9/sub:z:0normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_9/truediv?
category_encoding_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_9/Const?
category_encoding_9/MaxMaxAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0"category_encoding_9/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_9/Max?
category_encoding_9/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_9/Const_1?
category_encoding_9/MinMinAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0$category_encoding_9/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_9/Min{
category_encoding_9/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding_9/Cast/x?
category_encoding_9/CastCast#category_encoding_9/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_9/Cast?
category_encoding_9/GreaterGreatercategory_encoding_9/Cast:y:0 category_encoding_9/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_9/Greater~
category_encoding_9/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_9/Cast_1/x?
category_encoding_9/Cast_1Cast%category_encoding_9/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_9/Cast_1?
 category_encoding_9/GreaterEqualGreaterEqual category_encoding_9/Min:output:0category_encoding_9/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_9/GreaterEqual?
category_encoding_9/LogicalAnd
LogicalAndcategory_encoding_9/Greater:z:0$category_encoding_9/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_9/LogicalAnd?
 category_encoding_9/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2622"
 category_encoding_9/Assert/Const?
(category_encoding_9/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2622*
(category_encoding_9/Assert/Assert/data_0?
!category_encoding_9/Assert/AssertAssert"category_encoding_9/LogicalAnd:z:01category_encoding_9/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2#
!category_encoding_9/Assert/Assert?
"category_encoding_9/bincount/ShapeShapeAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_9/bincount/Shape?
"category_encoding_9/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_9/bincount/Const?
!category_encoding_9/bincount/ProdProd+category_encoding_9/bincount/Shape:output:0+category_encoding_9/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_9/bincount/Prod?
&category_encoding_9/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_9/bincount/Greater/y?
$category_encoding_9/bincount/GreaterGreater*category_encoding_9/bincount/Prod:output:0/category_encoding_9/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_9/bincount/Greater?
!category_encoding_9/bincount/CastCast(category_encoding_9/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_9/bincount/Cast?
$category_encoding_9/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_9/bincount/Const_1?
 category_encoding_9/bincount/MaxMaxAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_9/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_9/bincount/Max?
"category_encoding_9/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_9/bincount/add/y?
 category_encoding_9/bincount/addAddV2)category_encoding_9/bincount/Max:output:0+category_encoding_9/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_9/bincount/add?
 category_encoding_9/bincount/mulMul%category_encoding_9/bincount/Cast:y:0$category_encoding_9/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_9/bincount/mul?
&category_encoding_9/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_9/bincount/minlength?
$category_encoding_9/bincount/MaximumMaximum/category_encoding_9/bincount/minlength:output:0$category_encoding_9/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_9/bincount/Maximum?
&category_encoding_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_9/bincount/maxlength?
$category_encoding_9/bincount/MinimumMinimum/category_encoding_9/bincount/maxlength:output:0(category_encoding_9/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_9/bincount/Minimum?
$category_encoding_9/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_9/bincount/Const_2?
*category_encoding_9/bincount/DenseBincountDenseBincountAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_9/bincount/Minimum:z:0-category_encoding_9/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2,
*category_encoding_9/bincount/DenseBincountz
concatenate_19/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_19/concat/axis?
concatenate_19/concatConcatV2normalization_9/truediv:z:03category_encoding_9/bincount/DenseBincount:output:0#concatenate_19/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate_19/concat?
IdentityIdentityconcatenate_19/concat:output:0"^category_encoding_9/Assert/Assert'^normalization_9/Reshape/ReadVariableOp)^normalization_9/Reshape_1/ReadVariableOp9^string_lookup_9/None_lookup_table_find/LookupTableFindV2*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 2F
!category_encoding_9/Assert/Assert!category_encoding_9/Assert/Assert2P
&normalization_9/Reshape/ReadVariableOp&normalization_9/Reshape/ReadVariableOp2T
(normalization_9/Reshape_1/ReadVariableOp(normalization_9/Reshape_1/ReadVariableOp2t
8string_lookup_9/None_lookup_table_find/LookupTableFindV28string_lookup_9/None_lookup_table_find/LookupTableFindV2:T P
'
_output_shapes
:?????????
%
_user_specified_nameinputs/area:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/garden_area:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/month_sold:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/rooms:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/square_meter:d`
'
_output_shapes
:?????????
5
_user_specified_nameinputs/year_of_construction:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/year_sold:

_output_shapes
: 
?

?
.__inference_concatenate_18_layer_call_fn_53343
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_18_layer_call_and_return_conditional_losses_519132
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapest
r:?????????:?????????:?????????:?????????:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5
?	
?
C__inference_dense_38_layer_call_and_return_conditional_losses_52278

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
C__inference_model_19_layer_call_and_return_conditional_losses_52745
area
garden_area

month_sold	
rooms
square_meter
year_of_construction
	year_sold
model_18_52718
model_18_52720	
model_18_52722:
model_18_52724:&
sequential_9_52727:
??!
sequential_9_52729:	?%
sequential_9_52731:	?@ 
sequential_9_52733:@$
sequential_9_52735:@  
sequential_9_52737: $
sequential_9_52739:  
sequential_9_52741:
identity?? model_18/StatefulPartitionedCall?$sequential_9/StatefulPartitionedCall?
 model_18/StatefulPartitionedCallStatefulPartitionedCallareagarden_area
month_soldroomssquare_meteryear_of_construction	year_soldmodel_18_52718model_18_52720model_18_52722model_18_52724*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_18_layer_call_and_return_conditional_losses_519682"
 model_18/StatefulPartitionedCall?
$sequential_9/StatefulPartitionedCallStatefulPartitionedCall)model_18/StatefulPartitionedCall:output:0sequential_9_52727sequential_9_52729sequential_9_52731sequential_9_52733sequential_9_52735sequential_9_52737sequential_9_52739sequential_9_52741*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_523012&
$sequential_9/StatefulPartitionedCall?
IdentityIdentity-sequential_9/StatefulPartitionedCall:output:0!^model_18/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2D
 model_18/StatefulPartitionedCall model_18/StatefulPartitionedCall2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:M I
'
_output_shapes
:?????????

_user_specified_namearea:TP
'
_output_shapes
:?????????
%
_user_specified_namegarden_area:SO
'
_output_shapes
:?????????
$
_user_specified_name
month_sold:NJ
'
_output_shapes
:?????????

_user_specified_namerooms:UQ
'
_output_shapes
:?????????
&
_user_specified_namesquare_meter:]Y
'
_output_shapes
:?????????
.
_user_specified_nameyear_of_construction:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:

_output_shapes
: 
?
?
(__inference_model_19_layer_call_fn_52709
area
garden_area

month_sold	
rooms
square_meter
year_of_construction
	year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallareagarden_area
month_soldroomssquare_meteryear_of_construction	year_soldunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_19_layer_call_and_return_conditional_losses_526472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
'
_output_shapes
:?????????

_user_specified_namearea:TP
'
_output_shapes
:?????????
%
_user_specified_namegarden_area:SO
'
_output_shapes
:?????????
$
_user_specified_name
month_sold:NJ
'
_output_shapes
:?????????

_user_specified_namerooms:UQ
'
_output_shapes
:?????????
&
_user_specified_namesquare_meter:]Y
'
_output_shapes
:?????????
.
_user_specified_nameyear_of_construction:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:

_output_shapes
: 
?	
?
,__inference_sequential_9_layer_call_fn_53322

inputs
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_524072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
,__inference_sequential_9_layer_call_fn_52320
dense_36_input
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_36_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_523012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:??????????
(
_user_specified_namedense_36_input
?L
?
__inference__traced_save_53664
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop.
*savev2_dense_36_kernel_read_readvariableop,
(savev2_dense_36_bias_read_readvariableop.
*savev2_dense_37_kernel_read_readvariableop,
(savev2_dense_37_bias_read_readvariableop.
*savev2_dense_38_kernel_read_readvariableop,
(savev2_dense_38_bias_read_readvariableop.
*savev2_dense_39_kernel_read_readvariableop,
(savev2_dense_39_bias_read_readvariableop#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	U
Qsavev2_string_lookup_9_index_table_lookup_table_export_values_lookuptableexportv2W
Ssavev2_string_lookup_9_index_table_lookup_table_export_values_lookuptableexportv2_1	$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_36_kernel_m_read_readvariableop3
/savev2_adam_dense_36_bias_m_read_readvariableop5
1savev2_adam_dense_37_kernel_m_read_readvariableop3
/savev2_adam_dense_37_bias_m_read_readvariableop5
1savev2_adam_dense_38_kernel_m_read_readvariableop3
/savev2_adam_dense_38_bias_m_read_readvariableop5
1savev2_adam_dense_39_kernel_m_read_readvariableop3
/savev2_adam_dense_39_bias_m_read_readvariableop5
1savev2_adam_dense_36_kernel_v_read_readvariableop3
/savev2_adam_dense_36_bias_v_read_readvariableop5
1savev2_adam_dense_37_kernel_v_read_readvariableop3
/savev2_adam_dense_37_bias_v_read_readvariableop5
1savev2_adam_dense_38_kernel_v_read_readvariableop3
/savev2_adam_dense_38_bias_v_read_readvariableop5
1savev2_adam_dense_39_kernel_v_read_readvariableop3
/savev2_adam_dense_39_bias_v_read_readvariableop
savev2_const_1

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*?
value?B?%B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer_with_weights-0/_table/.ATTRIBUTES/table-keysBIlayer_with_weights-0/layer_with_weights-0/_table/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop*savev2_dense_36_kernel_read_readvariableop(savev2_dense_36_bias_read_readvariableop*savev2_dense_37_kernel_read_readvariableop(savev2_dense_37_bias_read_readvariableop*savev2_dense_38_kernel_read_readvariableop(savev2_dense_38_bias_read_readvariableop*savev2_dense_39_kernel_read_readvariableop(savev2_dense_39_bias_read_readvariableopsavev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableopQsavev2_string_lookup_9_index_table_lookup_table_export_values_lookuptableexportv2Ssavev2_string_lookup_9_index_table_lookup_table_export_values_lookuptableexportv2_1 savev2_total_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_36_kernel_m_read_readvariableop/savev2_adam_dense_36_bias_m_read_readvariableop1savev2_adam_dense_37_kernel_m_read_readvariableop/savev2_adam_dense_37_bias_m_read_readvariableop1savev2_adam_dense_38_kernel_m_read_readvariableop/savev2_adam_dense_38_bias_m_read_readvariableop1savev2_adam_dense_39_kernel_m_read_readvariableop/savev2_adam_dense_39_bias_m_read_readvariableop1savev2_adam_dense_36_kernel_v_read_readvariableop/savev2_adam_dense_36_bias_v_read_readvariableop1savev2_adam_dense_37_kernel_v_read_readvariableop/savev2_adam_dense_37_bias_v_read_readvariableop1savev2_adam_dense_38_kernel_v_read_readvariableop/savev2_adam_dense_38_bias_v_read_readvariableop1savev2_adam_dense_39_kernel_v_read_readvariableop/savev2_adam_dense_39_bias_v_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 *3
dtypes)
'2%			2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :
??:?:	?@:@:@ : : :::: ::: : :
??:?:	?@:@:@ : : ::
??:?:	?@:@:@ : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 	

_output_shapes
:@:$
 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?@:  

_output_shapes
:@:$! 

_output_shapes

:@ : "

_output_shapes
: :$# 

_output_shapes

: : $

_output_shapes
::%

_output_shapes
: 
?

?
C__inference_dense_36_layer_call_and_return_conditional_losses_53413

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
u
I__inference_concatenate_19_layer_call_and_return_conditional_losses_53396
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????:??????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?
Q
__inference__creator_53484
identity: ??string_lookup_9_index_table?
string_lookup_9_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_49019*
value_dtype0	2
string_lookup_9_index_table?
IdentityIdentity*string_lookup_9_index_table:table_handle:0^string_lookup_9_index_table*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2:
string_lookup_9_index_tablestring_lookup_9_index_table
?	
?
C__inference_dense_39_layer_call_and_return_conditional_losses_52294

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
C__inference_model_19_layer_call_and_return_conditional_losses_52541

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
model_18_52514
model_18_52516	
model_18_52518:
model_18_52520:&
sequential_9_52523:
??!
sequential_9_52525:	?%
sequential_9_52527:	?@ 
sequential_9_52529:@$
sequential_9_52531:@  
sequential_9_52533: $
sequential_9_52535:  
sequential_9_52537:
identity?? model_18/StatefulPartitionedCall?$sequential_9/StatefulPartitionedCall?
 model_18/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6model_18_52514model_18_52516model_18_52518model_18_52520*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_18_layer_call_and_return_conditional_losses_519682"
 model_18/StatefulPartitionedCall?
$sequential_9/StatefulPartitionedCallStatefulPartitionedCall)model_18/StatefulPartitionedCall:output:0sequential_9_52523sequential_9_52525sequential_9_52527sequential_9_52529sequential_9_52531sequential_9_52533sequential_9_52535sequential_9_52537*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_523012&
$sequential_9/StatefulPartitionedCall?
IdentityIdentity-sequential_9/StatefulPartitionedCall:output:0!^model_18/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2D
 model_18/StatefulPartitionedCall model_18/StatefulPartitionedCall2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?%
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_53251

inputs;
'dense_36_matmul_readvariableop_resource:
??7
(dense_36_biasadd_readvariableop_resource:	?:
'dense_37_matmul_readvariableop_resource:	?@6
(dense_37_biasadd_readvariableop_resource:@9
'dense_38_matmul_readvariableop_resource:@ 6
(dense_38_biasadd_readvariableop_resource: 9
'dense_39_matmul_readvariableop_resource: 6
(dense_39_biasadd_readvariableop_resource:
identity??dense_36/BiasAdd/ReadVariableOp?dense_36/MatMul/ReadVariableOp?dense_37/BiasAdd/ReadVariableOp?dense_37/MatMul/ReadVariableOp?dense_38/BiasAdd/ReadVariableOp?dense_38/MatMul/ReadVariableOp?dense_39/BiasAdd/ReadVariableOp?dense_39/MatMul/ReadVariableOp?
dense_36/MatMul/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_36/MatMul/ReadVariableOp?
dense_36/MatMulMatMulinputs&dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_36/MatMul?
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_36/BiasAdd/ReadVariableOp?
dense_36/BiasAddBiasAdddense_36/MatMul:product:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_36/BiasAddt
dense_36/TanhTanhdense_36/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_36/Tanh?
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02 
dense_37/MatMul/ReadVariableOp?
dense_37/MatMulMatMuldense_36/Tanh:y:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_37/MatMul?
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_37/BiasAdd/ReadVariableOp?
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_37/BiasAdd?
dense_38/MatMul/ReadVariableOpReadVariableOp'dense_38_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02 
dense_38/MatMul/ReadVariableOp?
dense_38/MatMulMatMuldense_37/BiasAdd:output:0&dense_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_38/MatMul?
dense_38/BiasAdd/ReadVariableOpReadVariableOp(dense_38_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_38/BiasAdd/ReadVariableOp?
dense_38/BiasAddBiasAdddense_38/MatMul:product:0'dense_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_38/BiasAdd?
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_39/MatMul/ReadVariableOp?
dense_39/MatMulMatMuldense_38/BiasAdd:output:0&dense_39/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_39/MatMul?
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_39/BiasAdd/ReadVariableOp?
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_39/BiasAdd?
IdentityIdentitydense_39/BiasAdd:output:0 ^dense_36/BiasAdd/ReadVariableOp^dense_36/MatMul/ReadVariableOp ^dense_37/BiasAdd/ReadVariableOp^dense_37/MatMul/ReadVariableOp ^dense_38/BiasAdd/ReadVariableOp^dense_38/MatMul/ReadVariableOp ^dense_39/BiasAdd/ReadVariableOp^dense_39/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2B
dense_36/BiasAdd/ReadVariableOpdense_36/BiasAdd/ReadVariableOp2@
dense_36/MatMul/ReadVariableOpdense_36/MatMul/ReadVariableOp2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2@
dense_37/MatMul/ReadVariableOpdense_37/MatMul/ReadVariableOp2B
dense_38/BiasAdd/ReadVariableOpdense_38/BiasAdd/ReadVariableOp2@
dense_38/MatMul/ReadVariableOpdense_38/MatMul/ReadVariableOp2B
dense_39/BiasAdd/ReadVariableOpdense_39/BiasAdd/ReadVariableOp2@
dense_39/MatMul/ReadVariableOpdense_39/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_52301

inputs"
dense_36_52247:
??
dense_36_52249:	?!
dense_37_52263:	?@
dense_37_52265:@ 
dense_38_52279:@ 
dense_38_52281:  
dense_39_52295: 
dense_39_52297:
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall? dense_38/StatefulPartitionedCall? dense_39/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCallinputsdense_36_52247dense_36_52249*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_36_layer_call_and_return_conditional_losses_522462"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_52263dense_37_52265*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_37_layer_call_and_return_conditional_losses_522622"
 dense_37/StatefulPartitionedCall?
 dense_38/StatefulPartitionedCallStatefulPartitionedCall)dense_37/StatefulPartitionedCall:output:0dense_38_52279dense_38_52281*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_38_layer_call_and_return_conditional_losses_522782"
 dense_38/StatefulPartitionedCall?
 dense_39/StatefulPartitionedCallStatefulPartitionedCall)dense_38/StatefulPartitionedCall:output:0dense_39_52295dense_39_52297*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_39_layer_call_and_return_conditional_losses_522942"
 dense_39/StatefulPartitionedCall?
IdentityIdentity)dense_39/StatefulPartitionedCall:output:0!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?U
?
C__inference_model_18_layer_call_and_return_conditional_losses_53124
inputs_area
inputs_garden_area
inputs_month_sold
inputs_rooms
inputs_square_meter
inputs_year_of_construction
inputs_year_soldI
Estring_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value	=
/normalization_9_reshape_readvariableop_resource:?
1normalization_9_reshape_1_readvariableop_resource:
identity??!category_encoding_9/Assert/Assert?&normalization_9/Reshape/ReadVariableOp?(normalization_9/Reshape_1/ReadVariableOp?8string_lookup_9/None_lookup_table_find/LookupTableFindV2?
8string_lookup_9/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleinputs_areaFstring_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_9/None_lookup_table_find/LookupTableFindV2z
concatenate_18/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_18/concat/axis?
concatenate_18/concatConcatV2inputs_roomsinputs_square_meterinputs_garden_areainputs_year_of_constructioninputs_year_soldinputs_month_sold#concatenate_18/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate_18/concat?
&normalization_9/Reshape/ReadVariableOpReadVariableOp/normalization_9_reshape_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization_9/Reshape/ReadVariableOp?
normalization_9/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_9/Reshape/shape?
normalization_9/ReshapeReshape.normalization_9/Reshape/ReadVariableOp:value:0&normalization_9/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_9/Reshape?
(normalization_9/Reshape_1/ReadVariableOpReadVariableOp1normalization_9_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02*
(normalization_9/Reshape_1/ReadVariableOp?
normalization_9/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_9/Reshape_1/shape?
normalization_9/Reshape_1Reshape0normalization_9/Reshape_1/ReadVariableOp:value:0(normalization_9/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_9/Reshape_1?
normalization_9/subSubconcatenate_18/concat:output:0 normalization_9/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization_9/sub?
normalization_9/SqrtSqrt"normalization_9/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_9/Sqrt{
normalization_9/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_9/Maximum/y?
normalization_9/MaximumMaximumnormalization_9/Sqrt:y:0"normalization_9/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_9/Maximum?
normalization_9/truedivRealDivnormalization_9/sub:z:0normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_9/truediv?
category_encoding_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_9/Const?
category_encoding_9/MaxMaxAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0"category_encoding_9/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_9/Max?
category_encoding_9/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_9/Const_1?
category_encoding_9/MinMinAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0$category_encoding_9/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_9/Min{
category_encoding_9/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding_9/Cast/x?
category_encoding_9/CastCast#category_encoding_9/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_9/Cast?
category_encoding_9/GreaterGreatercategory_encoding_9/Cast:y:0 category_encoding_9/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_9/Greater~
category_encoding_9/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_9/Cast_1/x?
category_encoding_9/Cast_1Cast%category_encoding_9/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_9/Cast_1?
 category_encoding_9/GreaterEqualGreaterEqual category_encoding_9/Min:output:0category_encoding_9/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_9/GreaterEqual?
category_encoding_9/LogicalAnd
LogicalAndcategory_encoding_9/Greater:z:0$category_encoding_9/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_9/LogicalAnd?
 category_encoding_9/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2622"
 category_encoding_9/Assert/Const?
(category_encoding_9/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2622*
(category_encoding_9/Assert/Assert/data_0?
!category_encoding_9/Assert/AssertAssert"category_encoding_9/LogicalAnd:z:01category_encoding_9/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2#
!category_encoding_9/Assert/Assert?
"category_encoding_9/bincount/ShapeShapeAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_9/bincount/Shape?
"category_encoding_9/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_9/bincount/Const?
!category_encoding_9/bincount/ProdProd+category_encoding_9/bincount/Shape:output:0+category_encoding_9/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_9/bincount/Prod?
&category_encoding_9/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_9/bincount/Greater/y?
$category_encoding_9/bincount/GreaterGreater*category_encoding_9/bincount/Prod:output:0/category_encoding_9/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_9/bincount/Greater?
!category_encoding_9/bincount/CastCast(category_encoding_9/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_9/bincount/Cast?
$category_encoding_9/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_9/bincount/Const_1?
 category_encoding_9/bincount/MaxMaxAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_9/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_9/bincount/Max?
"category_encoding_9/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_9/bincount/add/y?
 category_encoding_9/bincount/addAddV2)category_encoding_9/bincount/Max:output:0+category_encoding_9/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_9/bincount/add?
 category_encoding_9/bincount/mulMul%category_encoding_9/bincount/Cast:y:0$category_encoding_9/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_9/bincount/mul?
&category_encoding_9/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_9/bincount/minlength?
$category_encoding_9/bincount/MaximumMaximum/category_encoding_9/bincount/minlength:output:0$category_encoding_9/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_9/bincount/Maximum?
&category_encoding_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_9/bincount/maxlength?
$category_encoding_9/bincount/MinimumMinimum/category_encoding_9/bincount/maxlength:output:0(category_encoding_9/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_9/bincount/Minimum?
$category_encoding_9/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_9/bincount/Const_2?
*category_encoding_9/bincount/DenseBincountDenseBincountAstring_lookup_9/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_9/bincount/Minimum:z:0-category_encoding_9/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2,
*category_encoding_9/bincount/DenseBincountz
concatenate_19/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_19/concat/axis?
concatenate_19/concatConcatV2normalization_9/truediv:z:03category_encoding_9/bincount/DenseBincount:output:0#concatenate_19/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate_19/concat?
IdentityIdentityconcatenate_19/concat:output:0"^category_encoding_9/Assert/Assert'^normalization_9/Reshape/ReadVariableOp)^normalization_9/Reshape_1/ReadVariableOp9^string_lookup_9/None_lookup_table_find/LookupTableFindV2*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 2F
!category_encoding_9/Assert/Assert!category_encoding_9/Assert/Assert2P
&normalization_9/Reshape/ReadVariableOp&normalization_9/Reshape/ReadVariableOp2T
(normalization_9/Reshape_1/ReadVariableOp(normalization_9/Reshape_1/ReadVariableOp2t
8string_lookup_9/None_lookup_table_find/LookupTableFindV28string_lookup_9/None_lookup_table_find/LookupTableFindV2:T P
'
_output_shapes
:?????????
%
_user_specified_nameinputs/area:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/garden_area:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/month_sold:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/rooms:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/square_meter:d`
'
_output_shapes
:?????????
5
_user_specified_nameinputs/year_of_construction:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/year_sold:

_output_shapes
: 
?
?
(__inference_dense_39_layer_call_fn_53479

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_39_layer_call_and_return_conditional_losses_522942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
__inference_save_fn_53513
checkpoint_key[
Wstring_lookup_9_index_table_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	??Jstring_lookup_9_index_table_lookup_table_export_values/LookupTableExportV2?
Jstring_lookup_9_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Wstring_lookup_9_index_table_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::2L
Jstring_lookup_9_index_table_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1?
IdentityIdentityadd:z:0K^string_lookup_9_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const?

Identity_1IdentityConst:output:0K^string_lookup_9_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityQstring_lookup_9_index_table_lookup_table_export_values/LookupTableExportV2:keys:0K^string_lookup_9_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
:2

Identity_2?

Identity_3Identity	add_1:z:0K^string_lookup_9_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1?

Identity_4IdentityConst_1:output:0K^string_lookup_9_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentitySstring_lookup_9_index_table_lookup_table_export_values/LookupTableExportV2:values:0K^string_lookup_9_index_table_lookup_table_export_values/LookupTableExportV2*
T0	*
_output_shapes
:2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
Jstring_lookup_9_index_table_lookup_table_export_values/LookupTableExportV2Jstring_lookup_9_index_table_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?	
?
I__inference_concatenate_18_layer_call_and_return_conditional_losses_51913

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapest
r:?????????:?????????:?????????:?????????:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_model_18_layer_call_fn_53203
inputs_area
inputs_garden_area
inputs_month_sold
inputs_rooms
inputs_square_meter
inputs_year_of_construction
inputs_year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_areainputs_garden_areainputs_month_soldinputs_roomsinputs_square_meterinputs_year_of_constructioninputs_year_soldunknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_18_layer_call_and_return_conditional_losses_519682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_nameinputs/area:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/garden_area:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/month_sold:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/rooms:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/square_meter:d`
'
_output_shapes
:?????????
5
_user_specified_nameinputs/year_of_construction:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/year_sold:

_output_shapes
: 
?
?
(__inference_model_18_layer_call_fn_51979
area
garden_area

month_sold	
rooms
square_meter
year_of_construction
	year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallareagarden_area
month_soldroomssquare_meteryear_of_construction	year_soldunknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_18_layer_call_and_return_conditional_losses_519682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
'
_output_shapes
:?????????

_user_specified_namearea:TP
'
_output_shapes
:?????????
%
_user_specified_namegarden_area:SO
'
_output_shapes
:?????????
$
_user_specified_name
month_sold:NJ
'
_output_shapes
:?????????

_user_specified_namerooms:UQ
'
_output_shapes
:?????????
&
_user_specified_namesquare_meter:]Y
'
_output_shapes
:?????????
.
_user_specified_nameyear_of_construction:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:

_output_shapes
: 
?
,
__inference__destroyer_53494
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
??
?
!__inference__traced_restore_53779
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: 6
"assignvariableop_5_dense_36_kernel:
??/
 assignvariableop_6_dense_36_bias:	?5
"assignvariableop_7_dense_37_kernel:	?@.
 assignvariableop_8_dense_37_bias:@4
"assignvariableop_9_dense_38_kernel:@ /
!assignvariableop_10_dense_38_bias: 5
#assignvariableop_11_dense_39_kernel: /
!assignvariableop_12_dense_39_bias:&
assignvariableop_13_mean:*
assignvariableop_14_variance:#
assignvariableop_15_count:	 c
Ystring_lookup_9_index_table_table_restore_lookuptableimportv2_string_lookup_9_index_table: #
assignvariableop_16_total: %
assignvariableop_17_count_1: >
*assignvariableop_18_adam_dense_36_kernel_m:
??7
(assignvariableop_19_adam_dense_36_bias_m:	?=
*assignvariableop_20_adam_dense_37_kernel_m:	?@6
(assignvariableop_21_adam_dense_37_bias_m:@<
*assignvariableop_22_adam_dense_38_kernel_m:@ 6
(assignvariableop_23_adam_dense_38_bias_m: <
*assignvariableop_24_adam_dense_39_kernel_m: 6
(assignvariableop_25_adam_dense_39_bias_m:>
*assignvariableop_26_adam_dense_36_kernel_v:
??7
(assignvariableop_27_adam_dense_36_bias_v:	?=
*assignvariableop_28_adam_dense_37_kernel_v:	?@6
(assignvariableop_29_adam_dense_37_bias_v:@<
*assignvariableop_30_adam_dense_38_kernel_v:@ 6
(assignvariableop_31_adam_dense_38_bias_v: <
*assignvariableop_32_adam_dense_39_kernel_v: 6
(assignvariableop_33_adam_dense_39_bias_v:
identity_35??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?=string_lookup_9_index_table_table_restore/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*?
value?B?%B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer_with_weights-0/_table/.ATTRIBUTES/table-keysBIlayer_with_weights-0/layer_with_weights-0/_table/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%			2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_36_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense_36_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_37_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp assignvariableop_8_dense_37_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp"assignvariableop_9_dense_38_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp!assignvariableop_10_dense_38_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp#assignvariableop_11_dense_39_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_dense_39_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_15?
=string_lookup_9_index_table_table_restore/LookupTableImportV2LookupTableImportV2Ystring_lookup_9_index_table_table_restore_lookuptableimportv2_string_lookup_9_index_tableRestoreV2:tensors:16RestoreV2:tensors:17*	
Tin0*

Tout0	*.
_class$
" loc:@string_lookup_9_index_table*
_output_shapes
 2?
=string_lookup_9_index_table_table_restore/LookupTableImportV2n
Identity_16IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_dense_36_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_dense_36_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_37_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_dense_37_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_38_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_dense_38_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_dense_39_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_dense_39_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_dense_36_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_dense_36_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_dense_37_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_dense_37_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_dense_38_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_dense_38_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_dense_39_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_dense_39_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_339
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp>^string_lookup_9_index_table_table_restore/LookupTableImportV2"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_34?
Identity_35IdentityIdentity_34:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9>^string_lookup_9_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2
Identity_35"#
identity_35Identity_35:output:0*[
_input_shapesJ
H: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92~
=string_lookup_9_index_table_table_restore/LookupTableImportV2=string_lookup_9_index_table_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:40
.
_class$
" loc:@string_lookup_9_index_table
?
*
__inference_<lambda>_53526
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?	
?
C__inference_dense_38_layer_call_and_return_conditional_losses_53451

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_52824
area
garden_area

month_sold	
rooms
square_meter
year_of_construction
	year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallareagarden_area
month_soldroomssquare_meteryear_of_construction	year_soldunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_518802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
'
_output_shapes
:?????????

_user_specified_namearea:TP
'
_output_shapes
:?????????
%
_user_specified_namegarden_area:SO
'
_output_shapes
:?????????
$
_user_specified_name
month_sold:NJ
'
_output_shapes
:?????????

_user_specified_namerooms:UQ
'
_output_shapes
:?????????
&
_user_specified_namesquare_meter:]Y
'
_output_shapes
:?????????
.
_user_specified_nameyear_of_construction:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
5
area-
serving_default_area:0?????????
C
garden_area4
serving_default_garden_area:0?????????
A

month_sold3
serving_default_month_sold:0?????????
7
rooms.
serving_default_rooms:0?????????
E
square_meter5
serving_default_square_meter:0?????????
U
year_of_construction=
&serving_default_year_of_construction:0?????????
?
	year_sold2
serving_default_year_sold:0?????????@
sequential_90
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-0
layer-7
	layer_with_weights-1
	layer-8

	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"??
_tf_keras_network??{"name": "model_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_19", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}, "name": "area", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "garden_area"}, "name": "garden_area", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}, "name": "month_sold", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}, "name": "rooms", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}, "name": "square_meter", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_of_construction"}, "name": "year_of_construction", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}, "name": "year_sold", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "Functional", "config": {"name": "model_18", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}, "name": "rooms", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}, "name": "square_meter", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "garden_area"}, "name": "garden_area", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_of_construction"}, "name": "year_of_construction", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}, "name": "year_sold", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}, "name": "month_sold", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}, "name": "area", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Concatenate", "config": {"name": "concatenate_18", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_18", "inbound_nodes": [[["rooms", 0, 0, {}], ["square_meter", 0, 0, {}], ["garden_area", 0, 0, {}], ["year_of_construction", 0, 0, {}], ["year_sold", 0, 0, {}], ["month_sold", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_9", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "output_mode": "int", "pad_to_max_tokens": false, "vocabulary_size": 262, "vocabulary": null, "encoding": "utf-8"}, "name": "string_lookup_9", "inbound_nodes": [[["area", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "name": "normalization_9", "inbound_nodes": [[["concatenate_18", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_9", "trainable": true, "dtype": "float32", "num_tokens": 262, "output_mode": "binary", "sparse": false}, "name": "category_encoding_9", "inbound_nodes": [[["string_lookup_9", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_19", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_19", "inbound_nodes": [[["normalization_9", 0, 0, {}], ["category_encoding_9", 0, 0, {}]]]}], "input_layers": {"area": ["area", 0, 0], "rooms": ["rooms", 0, 0], "square_meter": ["square_meter", 0, 0], "garden_area": ["garden_area", 0, 0], "year_of_construction": ["year_of_construction", 0, 0], "year_sold": ["year_sold", 0, 0], "month_sold": ["month_sold", 0, 0]}, "output_layers": [["concatenate_19", 0, 0]]}, "name": "model_18", "inbound_nodes": [{"area": ["area", 0, 0, {}], "rooms": ["rooms", 0, 0, {}], "square_meter": ["square_meter", 0, 0, {}], "garden_area": ["garden_area", 0, 0, {}], "year_of_construction": ["year_of_construction", 0, 0, {}], "year_sold": ["year_sold", 0, 0, {}], "month_sold": ["month_sold", 0, 0, {}]}]}, {"class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 268]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_36_input"}}, {"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential_9", "inbound_nodes": [[["model_18", 1, 0, {}]]]}], "input_layers": {"area": ["area", 0, 0], "rooms": ["rooms", 0, 0], "square_meter": ["square_meter", 0, 0], "garden_area": ["garden_area", 0, 0], "year_of_construction": ["year_of_construction", 0, 0], "year_sold": ["year_sold", 0, 0], "month_sold": ["month_sold", 0, 0]}, "output_layers": [["sequential_9", 1, 0]]}, "shared_object_id": 27, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"area": {"class_name": "TensorShape", "items": [null, 1]}, "rooms": {"class_name": "TensorShape", "items": [null, 1]}, "square_meter": {"class_name": "TensorShape", "items": [null, 1]}, "garden_area": {"class_name": "TensorShape", "items": [null, 1]}, "year_of_construction": {"class_name": "TensorShape", "items": [null, 1]}, "year_sold": {"class_name": "TensorShape", "items": [null, 1]}, "month_sold": {"class_name": "TensorShape", "items": [null, 1]}}, "is_graph_network": true, "save_spec": {"area": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "string", "area"]}, "rooms": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "rooms"]}, "square_meter": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "square_meter"]}, "garden_area": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "garden_area"]}, "year_of_construction": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "year_of_construction"]}, "year_sold": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "year_sold"]}, "month_sold": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "month_sold"]}}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_19", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}, "name": "area", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "garden_area"}, "name": "garden_area", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}, "name": "month_sold", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}, "name": "rooms", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}, "name": "square_meter", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_of_construction"}, "name": "year_of_construction", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}, "name": "year_sold", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "Functional", "config": {"name": "model_18", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}, "name": "rooms", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}, "name": "square_meter", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "garden_area"}, "name": "garden_area", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_of_construction"}, "name": "year_of_construction", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}, "name": "year_sold", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}, "name": "month_sold", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}, "name": "area", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Concatenate", "config": {"name": "concatenate_18", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_18", "inbound_nodes": [[["rooms", 0, 0, {}], ["square_meter", 0, 0, {}], ["garden_area", 0, 0, {}], ["year_of_construction", 0, 0, {}], ["year_sold", 0, 0, {}], ["month_sold", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_9", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "output_mode": "int", "pad_to_max_tokens": false, "vocabulary_size": 262, "vocabulary": null, "encoding": "utf-8"}, "name": "string_lookup_9", "inbound_nodes": [[["area", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "name": "normalization_9", "inbound_nodes": [[["concatenate_18", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_9", "trainable": true, "dtype": "float32", "num_tokens": 262, "output_mode": "binary", "sparse": false}, "name": "category_encoding_9", "inbound_nodes": [[["string_lookup_9", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_19", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_19", "inbound_nodes": [[["normalization_9", 0, 0, {}], ["category_encoding_9", 0, 0, {}]]]}], "input_layers": {"area": ["area", 0, 0], "rooms": ["rooms", 0, 0], "square_meter": ["square_meter", 0, 0], "garden_area": ["garden_area", 0, 0], "year_of_construction": ["year_of_construction", 0, 0], "year_sold": ["year_sold", 0, 0], "month_sold": ["month_sold", 0, 0]}, "output_layers": [["concatenate_19", 0, 0]]}, "name": "model_18", "inbound_nodes": [{"area": ["area", 0, 0, {}], "rooms": ["rooms", 0, 0, {}], "square_meter": ["square_meter", 0, 0, {}], "garden_area": ["garden_area", 0, 0, {}], "year_of_construction": ["year_of_construction", 0, 0, {}], "year_sold": ["year_sold", 0, 0, {}], "month_sold": ["month_sold", 0, 0, {}]}], "shared_object_id": 12}, {"class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 268]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_36_input"}}, {"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential_9", "inbound_nodes": [[["model_18", 1, 0, {}]]], "shared_object_id": 26}], "input_layers": {"area": ["area", 0, 0], "rooms": ["rooms", 0, 0], "square_meter": ["square_meter", 0, 0], "garden_area": ["garden_area", 0, 0], "year_of_construction": ["year_of_construction", 0, 0], "year_sold": ["year_sold", 0, 0], "month_sold": ["month_sold", 0, 0]}, "output_layers": [["sequential_9", 1, 0]]}}, "training_config": {"loss": {"class_name": "MeanAbsoluteError", "config": {"reduction": "auto", "name": "mean_absolute_error"}, "shared_object_id": 35}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "area", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "garden_area", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "garden_area"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "month_sold", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "rooms", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "square_meter", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "year_of_construction", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_of_construction"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "year_sold", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}}
?X
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
layer_with_weights-0
layer-8
layer_with_weights-1
layer-9
layer-10
layer-11
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?U
_tf_keras_network?U{"name": "model_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_18", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}, "name": "rooms", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}, "name": "square_meter", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "garden_area"}, "name": "garden_area", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_of_construction"}, "name": "year_of_construction", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}, "name": "year_sold", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}, "name": "month_sold", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}, "name": "area", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Concatenate", "config": {"name": "concatenate_18", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_18", "inbound_nodes": [[["rooms", 0, 0, {}], ["square_meter", 0, 0, {}], ["garden_area", 0, 0, {}], ["year_of_construction", 0, 0, {}], ["year_sold", 0, 0, {}], ["month_sold", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_9", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "output_mode": "int", "pad_to_max_tokens": false, "vocabulary_size": 262, "vocabulary": null, "encoding": "utf-8"}, "name": "string_lookup_9", "inbound_nodes": [[["area", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "name": "normalization_9", "inbound_nodes": [[["concatenate_18", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_9", "trainable": true, "dtype": "float32", "num_tokens": 262, "output_mode": "binary", "sparse": false}, "name": "category_encoding_9", "inbound_nodes": [[["string_lookup_9", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_19", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_19", "inbound_nodes": [[["normalization_9", 0, 0, {}], ["category_encoding_9", 0, 0, {}]]]}], "input_layers": {"area": ["area", 0, 0], "rooms": ["rooms", 0, 0], "square_meter": ["square_meter", 0, 0], "garden_area": ["garden_area", 0, 0], "year_of_construction": ["year_of_construction", 0, 0], "year_sold": ["year_sold", 0, 0], "month_sold": ["month_sold", 0, 0]}, "output_layers": [["concatenate_19", 0, 0]]}, "inbound_nodes": [{"area": ["area", 0, 0, {}], "rooms": ["rooms", 0, 0, {}], "square_meter": ["square_meter", 0, 0, {}], "garden_area": ["garden_area", 0, 0, {}], "year_of_construction": ["year_of_construction", 0, 0, {}], "year_sold": ["year_sold", 0, 0, {}], "month_sold": ["month_sold", 0, 0, {}]}], "shared_object_id": 12, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"area": {"class_name": "TensorShape", "items": [null, 1]}, "rooms": {"class_name": "TensorShape", "items": [null, 1]}, "square_meter": {"class_name": "TensorShape", "items": [null, 1]}, "garden_area": {"class_name": "TensorShape", "items": [null, 1]}, "year_of_construction": {"class_name": "TensorShape", "items": [null, 1]}, "year_sold": {"class_name": "TensorShape", "items": [null, 1]}, "month_sold": {"class_name": "TensorShape", "items": [null, 1]}}, "is_graph_network": true, "save_spec": {"area": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "string", "area"]}, "rooms": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "rooms"]}, "square_meter": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "square_meter"]}, "garden_area": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "garden_area"]}, "year_of_construction": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "year_of_construction"]}, "year_sold": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "year_sold"]}, "month_sold": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "month_sold"]}}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_18", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}, "name": "rooms", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}, "name": "square_meter", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "garden_area"}, "name": "garden_area", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_of_construction"}, "name": "year_of_construction", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}, "name": "year_sold", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}, "name": "month_sold", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}, "name": "area", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Concatenate", "config": {"name": "concatenate_18", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_18", "inbound_nodes": [[["rooms", 0, 0, {}], ["square_meter", 0, 0, {}], ["garden_area", 0, 0, {}], ["year_of_construction", 0, 0, {}], ["year_sold", 0, 0, {}], ["month_sold", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "StringLookup", "config": {"name": "string_lookup_9", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "output_mode": "int", "pad_to_max_tokens": false, "vocabulary_size": 262, "vocabulary": null, "encoding": "utf-8"}, "name": "string_lookup_9", "inbound_nodes": [[["area", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "Normalization", "config": {"name": "normalization_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "name": "normalization_9", "inbound_nodes": [[["concatenate_18", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_9", "trainable": true, "dtype": "float32", "num_tokens": 262, "output_mode": "binary", "sparse": false}, "name": "category_encoding_9", "inbound_nodes": [[["string_lookup_9", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Concatenate", "config": {"name": "concatenate_19", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_19", "inbound_nodes": [[["normalization_9", 0, 0, {}], ["category_encoding_9", 0, 0, {}]]], "shared_object_id": 11}], "input_layers": {"area": ["area", 0, 0], "rooms": ["rooms", 0, 0], "square_meter": ["square_meter", 0, 0], "garden_area": ["garden_area", 0, 0], "year_of_construction": ["year_of_construction", 0, 0], "year_sold": ["year_sold", 0, 0], "month_sold": ["month_sold", 0, 0]}, "output_layers": [["concatenate_19", 0, 0]]}}}
?)
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
trainable_variables
regularization_losses
	variables
 	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?'
_tf_keras_sequential?'{"name": "sequential_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 268]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_36_input"}}, {"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "inbound_nodes": [[["model_18", 1, 0, {}]]], "shared_object_id": 26, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 268}}, "shared_object_id": 43}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 268]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 268]}, "float32", "dense_36_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 268]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_36_input"}, "shared_object_id": 13}, {"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 19}, {"class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 22}, {"class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 25}]}}}
?
!iter

"beta_1

#beta_2
	$decay
%learning_rate&m?'m?(m?)m?*m?+m?,m?-m?&v?'v?(v?)v?*v?+v?,v?-v?"
	optimizer
X
&0
'1
(2
)3
*4
+5
,6
-7"
trackable_list_wrapper
 "
trackable_list_wrapper
o
.1
/2
03
&4
'5
(6
)7
*8
+9
,10
-11"
trackable_list_wrapper
?
1metrics
trainable_variables
regularization_losses
2layer_regularization_losses
	variables
3non_trainable_variables

4layers
5layer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
6trainable_variables
7regularization_losses
8	variables
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "concatenate_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate_18", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["rooms", 0, 0, {}], ["square_meter", 0, 0, {}], ["garden_area", 0, 0, {}], ["year_of_construction", 0, 0, {}], ["year_sold", 0, 0, {}], ["month_sold", 0, 0, {}]]], "shared_object_id": 7, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}]}
?
:state_variables

;_table
<	keras_api"?
_tf_keras_layer?{"name": "string_lookup_9", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": null, "stateful": true, "must_restore_from_config": true, "class_name": "StringLookup", "config": {"name": "string_lookup_9", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "output_mode": "int", "pad_to_max_tokens": false, "vocabulary_size": 262, "vocabulary": null, "encoding": "utf-8"}, "inbound_nodes": [[["area", 0, 0, {}]]], "shared_object_id": 8}
?
=
_keep_axis
>_reduce_axis
?_reduce_axis_mask
@_broadcast_shape
.mean
/variance
	0count
A	keras_api
?_adapt_function"?
_tf_keras_layer?{"name": "normalization_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": true, "must_restore_from_config": true, "class_name": "Normalization", "config": {"name": "normalization_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "inbound_nodes": [[["concatenate_18", 0, 0, {}]]], "shared_object_id": 9, "build_input_shape": [null, 6]}
?
B	keras_api"?
_tf_keras_layer?{"name": "category_encoding_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "CategoryEncoding", "config": {"name": "category_encoding_9", "trainable": true, "dtype": "float32", "num_tokens": 262, "output_mode": "binary", "sparse": false}, "inbound_nodes": [[["string_lookup_9", 0, 0, {}]]], "shared_object_id": 10}
?
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "concatenate_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate_19", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["normalization_9", 0, 0, {}], ["category_encoding_9", 0, 0, {}]]], "shared_object_id": 11, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 6]}, {"class_name": "TensorShape", "items": [null, 262]}]}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
.1
/2
03"
trackable_list_wrapper
?
Gmetrics
trainable_variables
regularization_losses
Hlayer_regularization_losses
	variables
Inon_trainable_variables

Jlayers
Klayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

&kernel
'bias
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 268}}, "shared_object_id": 43}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 268]}}
?

(kernel
)bias
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 19, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

*kernel
+bias
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 45}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

,kernel
-bias
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
X
&0
'1
(2
)3
*4
+5
,6
-7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
&0
'1
(2
)3
*4
+5
,6
-7"
trackable_list_wrapper
?
\metrics
trainable_variables
regularization_losses
]layer_regularization_losses
	variables
^non_trainable_variables

_layers
`layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
#:!
??2dense_36/kernel
:?2dense_36/bias
": 	?@2dense_37/kernel
:@2dense_37/bias
!:@ 2dense_38/kernel
: 2dense_38/bias
!: 2dense_39/kernel
:2dense_39/bias
:2mean
:2variance
:	 2count
'
a0"
trackable_list_wrapper
 "
trackable_list_wrapper
5
.1
/2
03"
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
bmetrics
6trainable_variables
7regularization_losses
clayer_regularization_losses
8	variables
dnon_trainable_variables

elayers
flayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
T
?_create_resource
?_initialize
?_destroy_resourceR Z
table??
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
gmetrics
Ctrainable_variables
Dregularization_losses
hlayer_regularization_losses
E	variables
inon_trainable_variables

jlayers
klayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
.1
/2
03"
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
?
lmetrics
Ltrainable_variables
Mregularization_losses
mlayer_regularization_losses
N	variables
nnon_trainable_variables

olayers
player_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?
qmetrics
Ptrainable_variables
Qregularization_losses
rlayer_regularization_losses
R	variables
snon_trainable_variables

tlayers
ulayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
?
vmetrics
Ttrainable_variables
Uregularization_losses
wlayer_regularization_losses
V	variables
xnon_trainable_variables

ylayers
zlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
?
{metrics
Xtrainable_variables
Yregularization_losses
|layer_regularization_losses
Z	variables
}non_trainable_variables

~layers
layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 47}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
(:&
??2Adam/dense_36/kernel/m
!:?2Adam/dense_36/bias/m
':%	?@2Adam/dense_37/kernel/m
 :@2Adam/dense_37/bias/m
&:$@ 2Adam/dense_38/kernel/m
 : 2Adam/dense_38/bias/m
&:$ 2Adam/dense_39/kernel/m
 :2Adam/dense_39/bias/m
(:&
??2Adam/dense_36/kernel/v
!:?2Adam/dense_36/bias/v
':%	?@2Adam/dense_37/kernel/v
 :@2Adam/dense_37/bias/v
&:$@ 2Adam/dense_38/kernel/v
 : 2Adam/dense_38/bias/v
&:$ 2Adam/dense_39/kernel/v
 :2Adam/dense_39/bias/v
?2?
C__inference_model_19_layer_call_and_return_conditional_losses_52909
C__inference_model_19_layer_call_and_return_conditional_losses_52994
C__inference_model_19_layer_call_and_return_conditional_losses_52745
C__inference_model_19_layer_call_and_return_conditional_losses_52781?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_model_19_layer_call_fn_52568
(__inference_model_19_layer_call_fn_53029
(__inference_model_19_layer_call_fn_53064
(__inference_model_19_layer_call_fn_52709?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_51880?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *???
???
&
area?
area?????????
4
garden_area%?"
garden_area?????????
2

month_sold$?!

month_sold?????????
(
rooms?
rooms?????????
6
square_meter&?#
square_meter?????????
F
year_of_construction.?+
year_of_construction?????????
0
	year_sold#? 
	year_sold?????????
?2?
C__inference_model_18_layer_call_and_return_conditional_losses_53124
C__inference_model_18_layer_call_and_return_conditional_losses_53184
C__inference_model_18_layer_call_and_return_conditional_losses_52170
C__inference_model_18_layer_call_and_return_conditional_losses_52228?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_model_18_layer_call_fn_51979
(__inference_model_18_layer_call_fn_53203
(__inference_model_18_layer_call_fn_53222
(__inference_model_18_layer_call_fn_52112?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_sequential_9_layer_call_and_return_conditional_losses_53251
G__inference_sequential_9_layer_call_and_return_conditional_losses_53280
G__inference_sequential_9_layer_call_and_return_conditional_losses_52471
G__inference_sequential_9_layer_call_and_return_conditional_losses_52495?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_sequential_9_layer_call_fn_52320
,__inference_sequential_9_layer_call_fn_53301
,__inference_sequential_9_layer_call_fn_53322
,__inference_sequential_9_layer_call_fn_52447?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference_signature_wrapper_52824areagarden_area
month_soldroomssquare_meteryear_of_construction	year_sold"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_concatenate_18_layer_call_and_return_conditional_losses_53333?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_concatenate_18_layer_call_fn_53343?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_53389?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_concatenate_19_layer_call_and_return_conditional_losses_53396?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_concatenate_19_layer_call_fn_53402?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_36_layer_call_and_return_conditional_losses_53413?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_36_layer_call_fn_53422?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_37_layer_call_and_return_conditional_losses_53432?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_37_layer_call_fn_53441?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_38_layer_call_and_return_conditional_losses_53451?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_38_layer_call_fn_53460?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_39_layer_call_and_return_conditional_losses_53470?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_39_layer_call_fn_53479?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference__creator_53484?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_53489?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_53494?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference_save_fn_53513checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_53521restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
	J
Const6
__inference__creator_53484?

? 
? "? 8
__inference__destroyer_53494?

? 
? "? :
__inference__initializer_53489?

? 
? "? ?
 __inference__wrapped_model_51880?;?./&'()*+,-???
???
???
&
area?
area?????????
4
garden_area%?"
garden_area?????????
2

month_sold$?!

month_sold?????????
(
rooms?
rooms?????????
6
square_meter&?#
square_meter?????????
F
year_of_construction.?+
year_of_construction?????????
0
	year_sold#? 
	year_sold?????????
? ";?8
6
sequential_9&?#
sequential_9?????????l
__inference_adapt_step_53389L0./A?>
7?4
2?/?
??????????IteratorSpec
? "
 ?
I__inference_concatenate_18_layer_call_and_return_conditional_losses_53333????
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
? "%?"
?
0?????????
? ?
.__inference_concatenate_18_layer_call_fn_53343????
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
? "???????????
I__inference_concatenate_19_layer_call_and_return_conditional_losses_53396?[?X
Q?N
L?I
"?
inputs/0?????????
#? 
inputs/1??????????
? "&?#
?
0??????????
? ?
.__inference_concatenate_19_layer_call_fn_53402x[?X
Q?N
L?I
"?
inputs/0?????????
#? 
inputs/1??????????
? "????????????
C__inference_dense_36_layer_call_and_return_conditional_losses_53413^&'0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
(__inference_dense_36_layer_call_fn_53422Q&'0?-
&?#
!?
inputs??????????
? "????????????
C__inference_dense_37_layer_call_and_return_conditional_losses_53432]()0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? |
(__inference_dense_37_layer_call_fn_53441P()0?-
&?#
!?
inputs??????????
? "??????????@?
C__inference_dense_38_layer_call_and_return_conditional_losses_53451\*+/?,
%?"
 ?
inputs?????????@
? "%?"
?
0????????? 
? {
(__inference_dense_38_layer_call_fn_53460O*+/?,
%?"
 ?
inputs?????????@
? "?????????? ?
C__inference_dense_39_layer_call_and_return_conditional_losses_53470\,-/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? {
(__inference_dense_39_layer_call_fn_53479O,-/?,
%?"
 ?
inputs????????? 
? "???????????
C__inference_model_18_layer_call_and_return_conditional_losses_52170?;?./???
???
???
&
area?
area?????????
4
garden_area%?"
garden_area?????????
2

month_sold$?!

month_sold?????????
(
rooms?
rooms?????????
6
square_meter&?#
square_meter?????????
F
year_of_construction.?+
year_of_construction?????????
0
	year_sold#? 
	year_sold?????????
p 

 
? "&?#
?
0??????????
? ?
C__inference_model_18_layer_call_and_return_conditional_losses_52228?;?./???
???
???
&
area?
area?????????
4
garden_area%?"
garden_area?????????
2

month_sold$?!

month_sold?????????
(
rooms?
rooms?????????
6
square_meter&?#
square_meter?????????
F
year_of_construction.?+
year_of_construction?????????
0
	year_sold#? 
	year_sold?????????
p

 
? "&?#
?
0??????????
? ?
C__inference_model_18_layer_call_and_return_conditional_losses_53124?;?./???
???
???
-
area%?"
inputs/area?????????
;
garden_area,?)
inputs/garden_area?????????
9

month_sold+?(
inputs/month_sold?????????
/
rooms&?#
inputs/rooms?????????
=
square_meter-?*
inputs/square_meter?????????
M
year_of_construction5?2
inputs/year_of_construction?????????
7
	year_sold*?'
inputs/year_sold?????????
p 

 
? "&?#
?
0??????????
? ?
C__inference_model_18_layer_call_and_return_conditional_losses_53184?;?./???
???
???
-
area%?"
inputs/area?????????
;
garden_area,?)
inputs/garden_area?????????
9

month_sold+?(
inputs/month_sold?????????
/
rooms&?#
inputs/rooms?????????
=
square_meter-?*
inputs/square_meter?????????
M
year_of_construction5?2
inputs/year_of_construction?????????
7
	year_sold*?'
inputs/year_sold?????????
p

 
? "&?#
?
0??????????
? ?
(__inference_model_18_layer_call_fn_51979?;?./???
???
???
&
area?
area?????????
4
garden_area%?"
garden_area?????????
2

month_sold$?!

month_sold?????????
(
rooms?
rooms?????????
6
square_meter&?#
square_meter?????????
F
year_of_construction.?+
year_of_construction?????????
0
	year_sold#? 
	year_sold?????????
p 

 
? "????????????
(__inference_model_18_layer_call_fn_52112?;?./???
???
???
&
area?
area?????????
4
garden_area%?"
garden_area?????????
2

month_sold$?!

month_sold?????????
(
rooms?
rooms?????????
6
square_meter&?#
square_meter?????????
F
year_of_construction.?+
year_of_construction?????????
0
	year_sold#? 
	year_sold?????????
p

 
? "????????????
(__inference_model_18_layer_call_fn_53203?;?./???
???
???
-
area%?"
inputs/area?????????
;
garden_area,?)
inputs/garden_area?????????
9

month_sold+?(
inputs/month_sold?????????
/
rooms&?#
inputs/rooms?????????
=
square_meter-?*
inputs/square_meter?????????
M
year_of_construction5?2
inputs/year_of_construction?????????
7
	year_sold*?'
inputs/year_sold?????????
p 

 
? "????????????
(__inference_model_18_layer_call_fn_53222?;?./???
???
???
-
area%?"
inputs/area?????????
;
garden_area,?)
inputs/garden_area?????????
9

month_sold+?(
inputs/month_sold?????????
/
rooms&?#
inputs/rooms?????????
=
square_meter-?*
inputs/square_meter?????????
M
year_of_construction5?2
inputs/year_of_construction?????????
7
	year_sold*?'
inputs/year_sold?????????
p

 
? "????????????
C__inference_model_19_layer_call_and_return_conditional_losses_52745?;?./&'()*+,-???
???
???
&
area?
area?????????
4
garden_area%?"
garden_area?????????
2

month_sold$?!

month_sold?????????
(
rooms?
rooms?????????
6
square_meter&?#
square_meter?????????
F
year_of_construction.?+
year_of_construction?????????
0
	year_sold#? 
	year_sold?????????
p 

 
? "%?"
?
0?????????
? ?
C__inference_model_19_layer_call_and_return_conditional_losses_52781?;?./&'()*+,-???
???
???
&
area?
area?????????
4
garden_area%?"
garden_area?????????
2

month_sold$?!

month_sold?????????
(
rooms?
rooms?????????
6
square_meter&?#
square_meter?????????
F
year_of_construction.?+
year_of_construction?????????
0
	year_sold#? 
	year_sold?????????
p

 
? "%?"
?
0?????????
? ?
C__inference_model_19_layer_call_and_return_conditional_losses_52909?;?./&'()*+,-???
???
???
-
area%?"
inputs/area?????????
;
garden_area,?)
inputs/garden_area?????????
9

month_sold+?(
inputs/month_sold?????????
/
rooms&?#
inputs/rooms?????????
=
square_meter-?*
inputs/square_meter?????????
M
year_of_construction5?2
inputs/year_of_construction?????????
7
	year_sold*?'
inputs/year_sold?????????
p 

 
? "%?"
?
0?????????
? ?
C__inference_model_19_layer_call_and_return_conditional_losses_52994?;?./&'()*+,-???
???
???
-
area%?"
inputs/area?????????
;
garden_area,?)
inputs/garden_area?????????
9

month_sold+?(
inputs/month_sold?????????
/
rooms&?#
inputs/rooms?????????
=
square_meter-?*
inputs/square_meter?????????
M
year_of_construction5?2
inputs/year_of_construction?????????
7
	year_sold*?'
inputs/year_sold?????????
p

 
? "%?"
?
0?????????
? ?
(__inference_model_19_layer_call_fn_52568?;?./&'()*+,-???
???
???
&
area?
area?????????
4
garden_area%?"
garden_area?????????
2

month_sold$?!

month_sold?????????
(
rooms?
rooms?????????
6
square_meter&?#
square_meter?????????
F
year_of_construction.?+
year_of_construction?????????
0
	year_sold#? 
	year_sold?????????
p 

 
? "???????????
(__inference_model_19_layer_call_fn_52709?;?./&'()*+,-???
???
???
&
area?
area?????????
4
garden_area%?"
garden_area?????????
2

month_sold$?!

month_sold?????????
(
rooms?
rooms?????????
6
square_meter&?#
square_meter?????????
F
year_of_construction.?+
year_of_construction?????????
0
	year_sold#? 
	year_sold?????????
p

 
? "???????????
(__inference_model_19_layer_call_fn_53029?;?./&'()*+,-???
???
???
-
area%?"
inputs/area?????????
;
garden_area,?)
inputs/garden_area?????????
9

month_sold+?(
inputs/month_sold?????????
/
rooms&?#
inputs/rooms?????????
=
square_meter-?*
inputs/square_meter?????????
M
year_of_construction5?2
inputs/year_of_construction?????????
7
	year_sold*?'
inputs/year_sold?????????
p 

 
? "???????????
(__inference_model_19_layer_call_fn_53064?;?./&'()*+,-???
???
???
-
area%?"
inputs/area?????????
;
garden_area,?)
inputs/garden_area?????????
9

month_sold+?(
inputs/month_sold?????????
/
rooms&?#
inputs/rooms?????????
=
square_meter-?*
inputs/square_meter?????????
M
year_of_construction5?2
inputs/year_of_construction?????????
7
	year_sold*?'
inputs/year_sold?????????
p

 
? "??????????y
__inference_restore_fn_53521Y;K?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_53513?;&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
G__inference_sequential_9_layer_call_and_return_conditional_losses_52471s&'()*+,-@?=
6?3
)?&
dense_36_input??????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_9_layer_call_and_return_conditional_losses_52495s&'()*+,-@?=
6?3
)?&
dense_36_input??????????
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_9_layer_call_and_return_conditional_losses_53251k&'()*+,-8?5
.?+
!?
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_9_layer_call_and_return_conditional_losses_53280k&'()*+,-8?5
.?+
!?
inputs??????????
p

 
? "%?"
?
0?????????
? ?
,__inference_sequential_9_layer_call_fn_52320f&'()*+,-@?=
6?3
)?&
dense_36_input??????????
p 

 
? "???????????
,__inference_sequential_9_layer_call_fn_52447f&'()*+,-@?=
6?3
)?&
dense_36_input??????????
p

 
? "???????????
,__inference_sequential_9_layer_call_fn_53301^&'()*+,-8?5
.?+
!?
inputs??????????
p 

 
? "???????????
,__inference_sequential_9_layer_call_fn_53322^&'()*+,-8?5
.?+
!?
inputs??????????
p

 
? "???????????
#__inference_signature_wrapper_52824?;?./&'()*+,-???
? 
???
&
area?
area?????????
4
garden_area%?"
garden_area?????????
2

month_sold$?!

month_sold?????????
(
rooms?
rooms?????????
6
square_meter&?#
square_meter?????????
F
year_of_construction.?+
year_of_construction?????????
0
	year_sold#? 
	year_sold?????????";?8
6
sequential_9&?#
sequential_9?????????