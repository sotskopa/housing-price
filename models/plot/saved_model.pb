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
9
	IdentityN

input2T
output2T"
T
list(type)(0
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
0
Sigmoid
x"T
y"T"
Ttype:

2
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
{
dense_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	W?* 
shared_namedense_36/kernel
t
#dense_36/kernel/Read/ReadVariableOpReadVariableOpdense_36/kernel*
_output_shapes
:	W?*
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
shape:*
shared_namemean
Y
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
:*
dtype0
h
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance
a
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
:*
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
shared_nametable_34391*
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
dtype0*
shape:	W?*'
shared_nameAdam/dense_36/kernel/m
?
*Adam/dense_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_36/kernel/m*
_output_shapes
:	W?*
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
dtype0*
shape:	W?*'
shared_nameAdam/dense_36/kernel/v
?
*Adam/dense_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_36/kernel/v*
_output_shapes
:	W?*
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
__inference_<lambda>_37310

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
layer_with_weights-0
layer-6
layer_with_weights-1
layer-7
		optimizer

regularization_losses
trainable_variables
	variables
	keras_api

signatures
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
layer-4
layer-5
layer-6
layer_with_weights-0
layer-7
layer_with_weights-1
layer-8
layer-9
layer-10
regularization_losses
trainable_variables
	variables
	keras_api
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
regularization_losses
trainable_variables
	variables
	keras_api
?
 iter

!beta_1

"beta_2
	#decay
$learning_rate%m?&m?'m?(m?)m?*m?+m?,m?%v?&v?'v?(v?)v?*v?+v?,v?
 
8
%0
&1
'2
(3
)4
*5
+6
,7
O
-1
.2
/3
%4
&5
'6
(7
)8
*9
+10
,11
?
0non_trainable_variables
1layer_metrics

regularization_losses

2layers
3metrics
4layer_regularization_losses
trainable_variables
	variables
 
R
5regularization_losses
6trainable_variables
7	variables
8	keras_api
0
9state_variables

:_table
;	keras_api
?
<
_keep_axis
=_reduce_axis
>_reduce_axis_mask
?_broadcast_shape
-mean
.variance
	/count
@	keras_api

A	keras_api
R
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
 
 

-1
.2
/3
?
Fnon_trainable_variables
Glayer_metrics
regularization_losses

Hlayers
Imetrics
Jlayer_regularization_losses
trainable_variables
	variables
h

%kernel
&bias
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
h

'kernel
(bias
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
h

)kernel
*bias
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
h

+kernel
,bias
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
 
8
%0
&1
'2
(3
)4
*5
+6
,7
8
%0
&1
'2
(3
)4
*5
+6
,7
?
[non_trainable_variables
\layer_metrics
regularization_losses

]layers
^metrics
_layer_regularization_losses
trainable_variables
	variables
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

-1
.2
/3
 
8
0
1
2
3
4
5
6
7

`0
 
 
 
 
?
anon_trainable_variables
blayer_metrics
5regularization_losses

clayers
dmetrics
elayer_regularization_losses
6trainable_variables
7	variables
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
fnon_trainable_variables
glayer_metrics
Bregularization_losses

hlayers
imetrics
jlayer_regularization_losses
Ctrainable_variables
D	variables

-1
.2
/3
 
N
0
1
2
3
4
5
6
7
8
9
10
 
 
 

%0
&1

%0
&1
?
knon_trainable_variables
llayer_metrics
Kregularization_losses

mlayers
nmetrics
olayer_regularization_losses
Ltrainable_variables
M	variables
 

'0
(1

'0
(1
?
pnon_trainable_variables
qlayer_metrics
Oregularization_losses

rlayers
smetrics
tlayer_regularization_losses
Ptrainable_variables
Q	variables
 

)0
*1

)0
*1
?
unon_trainable_variables
vlayer_metrics
Sregularization_losses

wlayers
xmetrics
ylayer_regularization_losses
Ttrainable_variables
U	variables
 

+0
,1

+0
,1
?
znon_trainable_variables
{layer_metrics
Wregularization_losses

|layers
}metrics
~layer_regularization_losses
Xtrainable_variables
Y	variables
 
 

0
1
2
3
 
 
7
	total

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

0
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
|
serving_default_year_soldPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_areaserving_default_garden_areaserving_default_month_soldserving_default_roomsserving_default_square_meterserving_default_year_soldstring_lookup_9_index_tableConstmeanvariancedense_36/kerneldense_36/biasdense_37/kerneldense_37/biasdense_38/kerneldense_38/biasdense_39/kerneldense_39/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_36593
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
__inference__traced_save_37447
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
!__inference__traced_restore_37562??
?
?
(__inference_model_18_layer_call_fn_36875
inputs_area
inputs_garden_area
inputs_month_sold
inputs_rooms
inputs_square_meter
inputs_year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_areainputs_garden_areainputs_month_soldinputs_roomsinputs_square_meterinputs_year_soldunknown	unknown_0	unknown_1	unknown_2*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????W*$
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_18_layer_call_and_return_conditional_losses_358582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????W2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes|
z:?????????:?????????:?????????:?????????:?????????:?????????: : : : 22
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
_user_specified_nameinputs/square_meter:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/year_sold:

_output_shapes
: 
?T
?
C__inference_model_18_layer_call_and_return_conditional_losses_35944
area
garden_area

month_sold	
rooms
square_meter
	year_soldI
Estring_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value	=
/normalization_9_reshape_readvariableop_resource:?
1normalization_9_reshape_1_readvariableop_resource:
identity??!category_encoding_9/Assert/Assert?&normalization_9/Reshape/ReadVariableOp?(normalization_9/Reshape_1/ReadVariableOp?8string_lookup_9/None_lookup_table_find/LookupTableFindV2?
8string_lookup_9/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleareaFstring_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_9/None_lookup_table_find/LookupTableFindV2?
concatenate_18/PartitionedCallPartitionedCallroomssquare_metergarden_area	year_sold
month_sold*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_18_layer_call_and_return_conditional_losses_356932 
concatenate_18/PartitionedCall?
&normalization_9/Reshape/ReadVariableOpReadVariableOp/normalization_9_reshape_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization_9/Reshape/ReadVariableOp?
normalization_9/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_9/Reshape/shape?
normalization_9/ReshapeReshape.normalization_9/Reshape/ReadVariableOp:value:0&normalization_9/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_9/Reshape?
(normalization_9/Reshape_1/ReadVariableOpReadVariableOp1normalization_9_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02*
(normalization_9/Reshape_1/ReadVariableOp?
normalization_9/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_9/Reshape_1/shape?
normalization_9/Reshape_1Reshape0normalization_9/Reshape_1/ReadVariableOp:value:0(normalization_9/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_9/Reshape_1?
normalization_9/subSub'concatenate_18/PartitionedCall:output:0 normalization_9/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization_9/sub?
normalization_9/SqrtSqrt"normalization_9/Reshape_1:output:0*
T0*
_output_shapes

:2
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

:2
normalization_9/Maximum?
normalization_9/truedivRealDivnormalization_9/sub:z:0normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????2
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
category_encoding_9/Minz
category_encoding_9/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :R2
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
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=822"
 category_encoding_9/Assert/Const?
(category_encoding_9/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=822*
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
dtype0	*
value	B	 RR2(
&category_encoding_9/bincount/minlength?
$category_encoding_9/bincount/MaximumMaximum/category_encoding_9/bincount/minlength:output:0$category_encoding_9/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_9/bincount/Maximum?
&category_encoding_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 RR2(
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

Tidx0	*'
_output_shapes
:?????????R*
binary_output(2,
*category_encoding_9/bincount/DenseBincount?
concatenate_19/PartitionedCallPartitionedCallnormalization_9/truediv:z:03category_encoding_9/bincount/DenseBincount:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????W* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_19_layer_call_and_return_conditional_losses_357452 
concatenate_19/PartitionedCall?
IdentityIdentity'concatenate_19/PartitionedCall:output:0"^category_encoding_9/Assert/Assert'^normalization_9/Reshape/ReadVariableOp)^normalization_9/Reshape_1/ReadVariableOp9^string_lookup_9/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????W2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes|
z:?????????:?????????:?????????:?????????:?????????:?????????: : : : 2F
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
_user_specified_namesquare_meter:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:

_output_shapes
: 
?
Q
__inference__creator_37268
identity: ??string_lookup_9_index_table?
string_lookup_9_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_34391*
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
?
?
(__inference_model_19_layer_call_fn_36661
inputs_area
inputs_garden_area
inputs_month_sold
inputs_rooms
inputs_square_meter
inputs_year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
	unknown_3:	W?
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_areainputs_garden_areainputs_month_soldinputs_roomsinputs_square_meterinputs_year_soldunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_19_layer_call_and_return_conditional_losses_364202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 22
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
_user_specified_nameinputs/square_meter:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/year_sold:

_output_shapes
: 
?S
?
C__inference_model_18_layer_call_and_return_conditional_losses_35858

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5I
Estring_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value	=
/normalization_9_reshape_readvariableop_resource:?
1normalization_9_reshape_1_readvariableop_resource:
identity??!category_encoding_9/Assert/Assert?&normalization_9/Reshape/ReadVariableOp?(normalization_9/Reshape_1/ReadVariableOp?8string_lookup_9/None_lookup_table_find/LookupTableFindV2?
8string_lookup_9/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleinputsFstring_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_9/None_lookup_table_find/LookupTableFindV2?
concatenate_18/PartitionedCallPartitionedCallinputs_3inputs_4inputs_1inputs_5inputs_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_18_layer_call_and_return_conditional_losses_356932 
concatenate_18/PartitionedCall?
&normalization_9/Reshape/ReadVariableOpReadVariableOp/normalization_9_reshape_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization_9/Reshape/ReadVariableOp?
normalization_9/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_9/Reshape/shape?
normalization_9/ReshapeReshape.normalization_9/Reshape/ReadVariableOp:value:0&normalization_9/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_9/Reshape?
(normalization_9/Reshape_1/ReadVariableOpReadVariableOp1normalization_9_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02*
(normalization_9/Reshape_1/ReadVariableOp?
normalization_9/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_9/Reshape_1/shape?
normalization_9/Reshape_1Reshape0normalization_9/Reshape_1/ReadVariableOp:value:0(normalization_9/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_9/Reshape_1?
normalization_9/subSub'concatenate_18/PartitionedCall:output:0 normalization_9/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization_9/sub?
normalization_9/SqrtSqrt"normalization_9/Reshape_1:output:0*
T0*
_output_shapes

:2
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

:2
normalization_9/Maximum?
normalization_9/truedivRealDivnormalization_9/sub:z:0normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????2
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
category_encoding_9/Minz
category_encoding_9/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :R2
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
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=822"
 category_encoding_9/Assert/Const?
(category_encoding_9/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=822*
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
dtype0	*
value	B	 RR2(
&category_encoding_9/bincount/minlength?
$category_encoding_9/bincount/MaximumMaximum/category_encoding_9/bincount/minlength:output:0$category_encoding_9/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_9/bincount/Maximum?
&category_encoding_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 RR2(
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

Tidx0	*'
_output_shapes
:?????????R*
binary_output(2,
*category_encoding_9/bincount/DenseBincount?
concatenate_19/PartitionedCallPartitionedCallnormalization_9/truediv:z:03category_encoding_9/bincount/DenseBincount:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????W* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_19_layer_call_and_return_conditional_losses_357452 
concatenate_19/PartitionedCall?
IdentityIdentity'concatenate_19/PartitionedCall:output:0"^category_encoding_9/Assert/Assert'^normalization_9/Reshape/ReadVariableOp)^normalization_9/Reshape_1/ReadVariableOp9^string_lookup_9/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????W2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes|
z:?????????:?????????:?????????:?????????:?????????:?????????: : : : 2F
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
 
_user_specified_nameinputs:

_output_shapes
: 
?
u
I__inference_concatenate_19_layer_call_and_return_conditional_losses_37181
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
T0*'
_output_shapes
:?????????W2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????W2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????R:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????R
"
_user_specified_name
inputs/1
?
?
(__inference_model_18_layer_call_fn_36857
inputs_area
inputs_garden_area
inputs_month_sold
inputs_rooms
inputs_square_meter
inputs_year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_areainputs_garden_areainputs_month_soldinputs_roomsinputs_square_meterinputs_year_soldunknown	unknown_0	unknown_1	unknown_2*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????W*$
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_18_layer_call_and_return_conditional_losses_357482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????W2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes|
z:?????????:?????????:?????????:?????????:?????????:?????????: : : : 22
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
_user_specified_nameinputs/square_meter:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/year_sold:

_output_shapes
: 
?
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_36249
dense_36_input!
dense_36_36228:	W?
dense_36_36230:	?!
dense_37_36233:	?@
dense_37_36235:@ 
dense_38_36238:@ 
dense_38_36240:  
dense_39_36243: 
dense_39_36245:
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall? dense_38/StatefulPartitionedCall? dense_39/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCalldense_36_inputdense_36_36228dense_36_36230*
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
C__inference_dense_36_layer_call_and_return_conditional_losses_360242"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_36233dense_37_36235*
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
C__inference_dense_37_layer_call_and_return_conditional_losses_360402"
 dense_37/StatefulPartitionedCall?
 dense_38/StatefulPartitionedCallStatefulPartitionedCall)dense_37/StatefulPartitionedCall:output:0dense_38_36238dense_38_36240*
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
C__inference_dense_38_layer_call_and_return_conditional_losses_360562"
 dense_38/StatefulPartitionedCall?
 dense_39/StatefulPartitionedCallStatefulPartitionedCall)dense_38/StatefulPartitionedCall:output:0dense_39_36243dense_39_36245*
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
C__inference_dense_39_layer_call_and_return_conditional_losses_360722"
 dense_39/StatefulPartitionedCall?
IdentityIdentity)dense_39/StatefulPartitionedCall:output:0!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????W: : : : : : : : 2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall:W S
'
_output_shapes
:?????????W
(
_user_specified_namedense_36_input
?	
?
C__inference_dense_39_layer_call_and_return_conditional_losses_36072

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
(__inference_model_18_layer_call_fn_35887
area
garden_area

month_sold	
rooms
square_meter
	year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallareagarden_area
month_soldroomssquare_meter	year_soldunknown	unknown_0	unknown_1	unknown_2*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????W*$
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_18_layer_call_and_return_conditional_losses_358582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????W2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes|
z:?????????:?????????:?????????:?????????:?????????:?????????: : : : 22
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
_user_specified_namesquare_meter:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:

_output_shapes
: 
?
?
(__inference_model_19_layer_call_fn_36481
area
garden_area

month_sold	
rooms
square_meter
	year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
	unknown_3:	W?
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallareagarden_area
month_soldroomssquare_meter	year_soldunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_19_layer_call_and_return_conditional_losses_364202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 22
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
_user_specified_namesquare_meter:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:

_output_shapes
: 
?
?
(__inference_model_19_layer_call_fn_36344
area
garden_area

month_sold	
rooms
square_meter
	year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
	unknown_3:	W?
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallareagarden_area
month_soldroomssquare_meter	year_soldunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_19_layer_call_and_return_conditional_losses_363172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 22
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
_user_specified_namesquare_meter:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:

_output_shapes
: 
?	
?
C__inference_dense_37_layer_call_and_return_conditional_losses_36040

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
?
?
#__inference_signature_wrapper_36593
area
garden_area

month_sold	
rooms
square_meter
	year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
	unknown_3:	W?
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallareagarden_area
month_soldroomssquare_meter	year_soldunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_356632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 22
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
_user_specified_namesquare_meter:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:

_output_shapes
: 
?
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_36079

inputs!
dense_36_36025:	W?
dense_36_36027:	?!
dense_37_36041:	?@
dense_37_36043:@ 
dense_38_36057:@ 
dense_38_36059:  
dense_39_36073: 
dense_39_36075:
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall? dense_38/StatefulPartitionedCall? dense_39/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCallinputsdense_36_36025dense_36_36027*
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
C__inference_dense_36_layer_call_and_return_conditional_losses_360242"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_36041dense_37_36043*
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
C__inference_dense_37_layer_call_and_return_conditional_losses_360402"
 dense_37/StatefulPartitionedCall?
 dense_38/StatefulPartitionedCallStatefulPartitionedCall)dense_37/StatefulPartitionedCall:output:0dense_38_36057dense_38_36059*
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
C__inference_dense_38_layer_call_and_return_conditional_losses_360562"
 dense_38/StatefulPartitionedCall?
 dense_39/StatefulPartitionedCallStatefulPartitionedCall)dense_38/StatefulPartitionedCall:output:0dense_39_36073dense_39_36075*
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
C__inference_dense_39_layer_call_and_return_conditional_losses_360722"
 dense_39/StatefulPartitionedCall?
IdentityIdentity)dense_39/StatefulPartitionedCall:output:0!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????W: : : : : : : : 2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall:O K
'
_output_shapes
:?????????W
 
_user_specified_nameinputs
?S
?
C__inference_model_18_layer_call_and_return_conditional_losses_36934
inputs_area
inputs_garden_area
inputs_month_sold
inputs_rooms
inputs_square_meter
inputs_year_soldI
Estring_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value	=
/normalization_9_reshape_readvariableop_resource:?
1normalization_9_reshape_1_readvariableop_resource:
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
concatenate_18/concat/axis?
concatenate_18/concatConcatV2inputs_roomsinputs_square_meterinputs_garden_areainputs_year_soldinputs_month_sold#concatenate_18/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate_18/concat?
&normalization_9/Reshape/ReadVariableOpReadVariableOp/normalization_9_reshape_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization_9/Reshape/ReadVariableOp?
normalization_9/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_9/Reshape/shape?
normalization_9/ReshapeReshape.normalization_9/Reshape/ReadVariableOp:value:0&normalization_9/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_9/Reshape?
(normalization_9/Reshape_1/ReadVariableOpReadVariableOp1normalization_9_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02*
(normalization_9/Reshape_1/ReadVariableOp?
normalization_9/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_9/Reshape_1/shape?
normalization_9/Reshape_1Reshape0normalization_9/Reshape_1/ReadVariableOp:value:0(normalization_9/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_9/Reshape_1?
normalization_9/subSubconcatenate_18/concat:output:0 normalization_9/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization_9/sub?
normalization_9/SqrtSqrt"normalization_9/Reshape_1:output:0*
T0*
_output_shapes

:2
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

:2
normalization_9/Maximum?
normalization_9/truedivRealDivnormalization_9/sub:z:0normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????2
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
category_encoding_9/Minz
category_encoding_9/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :R2
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
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=822"
 category_encoding_9/Assert/Const?
(category_encoding_9/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=822*
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
dtype0	*
value	B	 RR2(
&category_encoding_9/bincount/minlength?
$category_encoding_9/bincount/MaximumMaximum/category_encoding_9/bincount/minlength:output:0$category_encoding_9/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_9/bincount/Maximum?
&category_encoding_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 RR2(
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

Tidx0	*'
_output_shapes
:?????????R*
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
T0*'
_output_shapes
:?????????W2
concatenate_19/concat?
IdentityIdentityconcatenate_19/concat:output:0"^category_encoding_9/Assert/Assert'^normalization_9/Reshape/ReadVariableOp)^normalization_9/Reshape_1/ReadVariableOp9^string_lookup_9/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????W2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes|
z:?????????:?????????:?????????:?????????:?????????:?????????: : : : 2F
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
_user_specified_nameinputs/square_meter:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/year_sold:

_output_shapes
: 
?	
?
__inference_restore_fn_37305
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
?
.
__inference__initializer_37273
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
?
Z
.__inference_concatenate_19_layer_call_fn_37174
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????W* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_19_layer_call_and_return_conditional_losses_357452
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????W2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????R:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????R
"
_user_specified_name
inputs/1
А
?
C__inference_model_19_layer_call_and_return_conditional_losses_36750
inputs_area
inputs_garden_area
inputs_month_sold
inputs_rooms
inputs_square_meter
inputs_year_soldR
Nmodel_18_string_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleS
Omodel_18_string_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value	F
8model_18_normalization_9_reshape_readvariableop_resource:H
:model_18_normalization_9_reshape_1_readvariableop_resource:G
4sequential_9_dense_36_matmul_readvariableop_resource:	W?D
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
model_18/concatenate_18/concatConcatV2inputs_roomsinputs_square_meterinputs_garden_areainputs_year_soldinputs_month_sold,model_18/concatenate_18/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2 
model_18/concatenate_18/concat?
/model_18/normalization_9/Reshape/ReadVariableOpReadVariableOp8model_18_normalization_9_reshape_readvariableop_resource*
_output_shapes
:*
dtype021
/model_18/normalization_9/Reshape/ReadVariableOp?
&model_18/normalization_9/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2(
&model_18/normalization_9/Reshape/shape?
 model_18/normalization_9/ReshapeReshape7model_18/normalization_9/Reshape/ReadVariableOp:value:0/model_18/normalization_9/Reshape/shape:output:0*
T0*
_output_shapes

:2"
 model_18/normalization_9/Reshape?
1model_18/normalization_9/Reshape_1/ReadVariableOpReadVariableOp:model_18_normalization_9_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype023
1model_18/normalization_9/Reshape_1/ReadVariableOp?
(model_18/normalization_9/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2*
(model_18/normalization_9/Reshape_1/shape?
"model_18/normalization_9/Reshape_1Reshape9model_18/normalization_9/Reshape_1/ReadVariableOp:value:01model_18/normalization_9/Reshape_1/shape:output:0*
T0*
_output_shapes

:2$
"model_18/normalization_9/Reshape_1?
model_18/normalization_9/subSub'model_18/concatenate_18/concat:output:0)model_18/normalization_9/Reshape:output:0*
T0*'
_output_shapes
:?????????2
model_18/normalization_9/sub?
model_18/normalization_9/SqrtSqrt+model_18/normalization_9/Reshape_1:output:0*
T0*
_output_shapes

:2
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

:2"
 model_18/normalization_9/Maximum?
 model_18/normalization_9/truedivRealDiv model_18/normalization_9/sub:z:0$model_18/normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????2"
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
dtype0*
value	B :R2%
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
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=822+
)model_18/category_encoding_9/Assert/Const?
1model_18/category_encoding_9/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=8223
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
dtype0	*
value	B	 RR21
/model_18/category_encoding_9/bincount/minlength?
-model_18/category_encoding_9/bincount/MaximumMaximum8model_18/category_encoding_9/bincount/minlength:output:0-model_18/category_encoding_9/bincount/mul:z:0*
T0	*
_output_shapes
: 2/
-model_18/category_encoding_9/bincount/Maximum?
/model_18/category_encoding_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 RR21
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

Tidx0	*'
_output_shapes
:?????????R*
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
T0*'
_output_shapes
:?????????W2 
model_18/concatenate_19/concat?
+sequential_9/dense_36/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_36_matmul_readvariableop_resource*
_output_shapes
:	W?*
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
sequential_9/dense_36/SigmoidSigmoid&sequential_9/dense_36/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_36/Sigmoid?
sequential_9/dense_36/mulMul&sequential_9/dense_36/BiasAdd:output:0!sequential_9/dense_36/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_36/mul?
sequential_9/dense_36/IdentityIdentitysequential_9/dense_36/mul:z:0*
T0*(
_output_shapes
:??????????2 
sequential_9/dense_36/Identity?
sequential_9/dense_36/IdentityN	IdentityNsequential_9/dense_36/mul:z:0&sequential_9/dense_36/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-36725*<
_output_shapes*
(:??????????:??????????2!
sequential_9/dense_36/IdentityN?
+sequential_9/dense_37/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_37_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02-
+sequential_9/dense_37/MatMul/ReadVariableOp?
sequential_9/dense_37/MatMulMatMul(sequential_9/dense_36/IdentityN:output:03sequential_9/dense_37/MatMul/ReadVariableOp:value:0*
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
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2X
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
_user_specified_nameinputs/square_meter:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/year_sold:

_output_shapes
: 
?,
?
__inference_adapt_step_37168
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceIteratorGetNext:components:0moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
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

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype02
ReadVariableOpW
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:2
mula
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:2
mul_1P
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:2
add_1x
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1_
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:2
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
:2
powz
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype02
ReadVariableOp_2_
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:2
add_2N
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:2
mul_2_
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:2
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
:2
pow_1c
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:2
add_3R
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:2
mul_3R
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:2
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
?	
?
C__inference_dense_38_layer_call_and_return_conditional_losses_37244

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
?
?
__inference_save_fn_37297
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
?
,__inference_sequential_9_layer_call_fn_37035

inputs
unknown:	W?
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_361852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????W: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????W
 
_user_specified_nameinputs
?
,
__inference__destroyer_37278
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
?
*
__inference_<lambda>_37310
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
?
?
C__inference_model_19_layer_call_and_return_conditional_losses_36516
area
garden_area

month_sold	
rooms
square_meter
	year_sold
model_18_36489
model_18_36491	
model_18_36493:
model_18_36495:%
sequential_9_36498:	W?!
sequential_9_36500:	?%
sequential_9_36502:	?@ 
sequential_9_36504:@$
sequential_9_36506:@  
sequential_9_36508: $
sequential_9_36510:  
sequential_9_36512:
identity?? model_18/StatefulPartitionedCall?$sequential_9/StatefulPartitionedCall?
 model_18/StatefulPartitionedCallStatefulPartitionedCallareagarden_area
month_soldroomssquare_meter	year_soldmodel_18_36489model_18_36491model_18_36493model_18_36495*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????W*$
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_18_layer_call_and_return_conditional_losses_357482"
 model_18/StatefulPartitionedCall?
$sequential_9/StatefulPartitionedCallStatefulPartitionedCall)model_18/StatefulPartitionedCall:output:0sequential_9_36498sequential_9_36500sequential_9_36502sequential_9_36504sequential_9_36506sequential_9_36508sequential_9_36510sequential_9_36512*
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_360792&
$sequential_9/StatefulPartitionedCall?
IdentityIdentity-sequential_9/StatefulPartitionedCall:output:0!^model_18/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2D
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
_user_specified_namesquare_meter:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:

_output_shapes
: 
?
?
(__inference_dense_36_layer_call_fn_37190

inputs
unknown:	W?
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
C__inference_dense_36_layer_call_and_return_conditional_losses_360242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????W: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????W
 
_user_specified_nameinputs
?L
?
__inference__traced_save_37447
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
?: : : : : : :	W?:?:	?@:@:@ : : :::: ::: : :	W?:?:	?@:@:@ : : ::	W?:?:	?@:@:@ : : :: 2(
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
: :%!

_output_shapes
:	W?:!
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
:: 

_output_shapes
::
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
: :%!

_output_shapes
:	W?:!
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
::%!

_output_shapes
:	W?:!
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
C__inference_dense_39_layer_call_and_return_conditional_losses_37263

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
?
?
I__inference_concatenate_18_layer_call_and_return_conditional_losses_37122
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:?????????:?????????:?????????:?????????:?????????:Q M
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
inputs/4
?	
?
.__inference_concatenate_18_layer_call_fn_37112
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_18_layer_call_and_return_conditional_losses_356932
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:?????????:?????????:?????????:?????????:?????????:Q M
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
inputs/4
?S
?
C__inference_model_18_layer_call_and_return_conditional_losses_36993
inputs_area
inputs_garden_area
inputs_month_sold
inputs_rooms
inputs_square_meter
inputs_year_soldI
Estring_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value	=
/normalization_9_reshape_readvariableop_resource:?
1normalization_9_reshape_1_readvariableop_resource:
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
concatenate_18/concat/axis?
concatenate_18/concatConcatV2inputs_roomsinputs_square_meterinputs_garden_areainputs_year_soldinputs_month_sold#concatenate_18/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate_18/concat?
&normalization_9/Reshape/ReadVariableOpReadVariableOp/normalization_9_reshape_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization_9/Reshape/ReadVariableOp?
normalization_9/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_9/Reshape/shape?
normalization_9/ReshapeReshape.normalization_9/Reshape/ReadVariableOp:value:0&normalization_9/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_9/Reshape?
(normalization_9/Reshape_1/ReadVariableOpReadVariableOp1normalization_9_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02*
(normalization_9/Reshape_1/ReadVariableOp?
normalization_9/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_9/Reshape_1/shape?
normalization_9/Reshape_1Reshape0normalization_9/Reshape_1/ReadVariableOp:value:0(normalization_9/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_9/Reshape_1?
normalization_9/subSubconcatenate_18/concat:output:0 normalization_9/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization_9/sub?
normalization_9/SqrtSqrt"normalization_9/Reshape_1:output:0*
T0*
_output_shapes

:2
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

:2
normalization_9/Maximum?
normalization_9/truedivRealDivnormalization_9/sub:z:0normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????2
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
category_encoding_9/Minz
category_encoding_9/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :R2
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
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=822"
 category_encoding_9/Assert/Const?
(category_encoding_9/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=822*
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
dtype0	*
value	B	 RR2(
&category_encoding_9/bincount/minlength?
$category_encoding_9/bincount/MaximumMaximum/category_encoding_9/bincount/minlength:output:0$category_encoding_9/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_9/bincount/Maximum?
&category_encoding_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 RR2(
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

Tidx0	*'
_output_shapes
:?????????R*
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
T0*'
_output_shapes
:?????????W2
concatenate_19/concat?
IdentityIdentityconcatenate_19/concat:output:0"^category_encoding_9/Assert/Assert'^normalization_9/Reshape/ReadVariableOp)^normalization_9/Reshape_1/ReadVariableOp9^string_lookup_9/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????W2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes|
z:?????????:?????????:?????????:?????????:?????????:?????????: : : : 2F
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
_user_specified_nameinputs/square_meter:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/year_sold:

_output_shapes
: 
?
?
(__inference_dense_38_layer_call_fn_37234

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
C__inference_dense_38_layer_call_and_return_conditional_losses_360562
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
?
?
(__inference_model_19_layer_call_fn_36627
inputs_area
inputs_garden_area
inputs_month_sold
inputs_rooms
inputs_square_meter
inputs_year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
	unknown_3:	W?
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_areainputs_garden_areainputs_month_soldinputs_roomsinputs_square_meterinputs_year_soldunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_19_layer_call_and_return_conditional_losses_363172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 22
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
_user_specified_nameinputs/square_meter:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/year_sold:

_output_shapes
: 
?
?
C__inference_dense_36_layer_call_and_return_conditional_losses_37206

inputs1
matmul_readvariableop_resource:	W?.
biasadd_readvariableop_resource:	?

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	W?*
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
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidc
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
mul\
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:??????????2

Identity?
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-37199*<
_output_shapes*
(:??????????:??????????2
	IdentityN?

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????W: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????W
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_35663
area
garden_area

month_sold	
rooms
square_meter
	year_sold[
Wmodel_19_model_18_string_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handle\
Xmodel_19_model_18_string_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value	O
Amodel_19_model_18_normalization_9_reshape_readvariableop_resource:Q
Cmodel_19_model_18_normalization_9_reshape_1_readvariableop_resource:P
=model_19_sequential_9_dense_36_matmul_readvariableop_resource:	W?M
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
'model_19/model_18/concatenate_18/concatConcatV2roomssquare_metergarden_area	year_sold
month_sold5model_19/model_18/concatenate_18/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2)
'model_19/model_18/concatenate_18/concat?
8model_19/model_18/normalization_9/Reshape/ReadVariableOpReadVariableOpAmodel_19_model_18_normalization_9_reshape_readvariableop_resource*
_output_shapes
:*
dtype02:
8model_19/model_18/normalization_9/Reshape/ReadVariableOp?
/model_19/model_18/normalization_9/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      21
/model_19/model_18/normalization_9/Reshape/shape?
)model_19/model_18/normalization_9/ReshapeReshape@model_19/model_18/normalization_9/Reshape/ReadVariableOp:value:08model_19/model_18/normalization_9/Reshape/shape:output:0*
T0*
_output_shapes

:2+
)model_19/model_18/normalization_9/Reshape?
:model_19/model_18/normalization_9/Reshape_1/ReadVariableOpReadVariableOpCmodel_19_model_18_normalization_9_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02<
:model_19/model_18/normalization_9/Reshape_1/ReadVariableOp?
1model_19/model_18/normalization_9/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      23
1model_19/model_18/normalization_9/Reshape_1/shape?
+model_19/model_18/normalization_9/Reshape_1ReshapeBmodel_19/model_18/normalization_9/Reshape_1/ReadVariableOp:value:0:model_19/model_18/normalization_9/Reshape_1/shape:output:0*
T0*
_output_shapes

:2-
+model_19/model_18/normalization_9/Reshape_1?
%model_19/model_18/normalization_9/subSub0model_19/model_18/concatenate_18/concat:output:02model_19/model_18/normalization_9/Reshape:output:0*
T0*'
_output_shapes
:?????????2'
%model_19/model_18/normalization_9/sub?
&model_19/model_18/normalization_9/SqrtSqrt4model_19/model_18/normalization_9/Reshape_1:output:0*
T0*
_output_shapes

:2(
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

:2+
)model_19/model_18/normalization_9/Maximum?
)model_19/model_18/normalization_9/truedivRealDiv)model_19/model_18/normalization_9/sub:z:0-model_19/model_18/normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????2+
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
dtype0*
value	B :R2.
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
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=8224
2model_19/model_18/category_encoding_9/Assert/Const?
:model_19/model_18/category_encoding_9/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=822<
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
dtype0	*
value	B	 RR2:
8model_19/model_18/category_encoding_9/bincount/minlength?
6model_19/model_18/category_encoding_9/bincount/MaximumMaximumAmodel_19/model_18/category_encoding_9/bincount/minlength:output:06model_19/model_18/category_encoding_9/bincount/mul:z:0*
T0	*
_output_shapes
: 28
6model_19/model_18/category_encoding_9/bincount/Maximum?
8model_19/model_18/category_encoding_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 RR2:
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

Tidx0	*'
_output_shapes
:?????????R*
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
T0*'
_output_shapes
:?????????W2)
'model_19/model_18/concatenate_19/concat?
4model_19/sequential_9/dense_36/MatMul/ReadVariableOpReadVariableOp=model_19_sequential_9_dense_36_matmul_readvariableop_resource*
_output_shapes
:	W?*
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
&model_19/sequential_9/dense_36/SigmoidSigmoid/model_19/sequential_9/dense_36/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&model_19/sequential_9/dense_36/Sigmoid?
"model_19/sequential_9/dense_36/mulMul/model_19/sequential_9/dense_36/BiasAdd:output:0*model_19/sequential_9/dense_36/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2$
"model_19/sequential_9/dense_36/mul?
'model_19/sequential_9/dense_36/IdentityIdentity&model_19/sequential_9/dense_36/mul:z:0*
T0*(
_output_shapes
:??????????2)
'model_19/sequential_9/dense_36/Identity?
(model_19/sequential_9/dense_36/IdentityN	IdentityN&model_19/sequential_9/dense_36/mul:z:0/model_19/sequential_9/dense_36/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-35638*<
_output_shapes*
(:??????????:??????????2*
(model_19/sequential_9/dense_36/IdentityN?
4model_19/sequential_9/dense_37/MatMul/ReadVariableOpReadVariableOp=model_19_sequential_9_dense_37_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype026
4model_19/sequential_9/dense_37/MatMul/ReadVariableOp?
%model_19/sequential_9/dense_37/MatMulMatMul1model_19/sequential_9/dense_36/IdentityN:output:0<model_19/sequential_9/dense_37/MatMul/ReadVariableOp:value:0*
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
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2j
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
_user_specified_namesquare_meter:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:

_output_shapes
: 
?	
?
,__inference_sequential_9_layer_call_fn_36098
dense_36_input
unknown:	W?
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_360792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????W: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????W
(
_user_specified_namedense_36_input
?S
?
C__inference_model_18_layer_call_and_return_conditional_losses_35748

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5I
Estring_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value	=
/normalization_9_reshape_readvariableop_resource:?
1normalization_9_reshape_1_readvariableop_resource:
identity??!category_encoding_9/Assert/Assert?&normalization_9/Reshape/ReadVariableOp?(normalization_9/Reshape_1/ReadVariableOp?8string_lookup_9/None_lookup_table_find/LookupTableFindV2?
8string_lookup_9/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleinputsFstring_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_9/None_lookup_table_find/LookupTableFindV2?
concatenate_18/PartitionedCallPartitionedCallinputs_3inputs_4inputs_1inputs_5inputs_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_18_layer_call_and_return_conditional_losses_356932 
concatenate_18/PartitionedCall?
&normalization_9/Reshape/ReadVariableOpReadVariableOp/normalization_9_reshape_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization_9/Reshape/ReadVariableOp?
normalization_9/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_9/Reshape/shape?
normalization_9/ReshapeReshape.normalization_9/Reshape/ReadVariableOp:value:0&normalization_9/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_9/Reshape?
(normalization_9/Reshape_1/ReadVariableOpReadVariableOp1normalization_9_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02*
(normalization_9/Reshape_1/ReadVariableOp?
normalization_9/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_9/Reshape_1/shape?
normalization_9/Reshape_1Reshape0normalization_9/Reshape_1/ReadVariableOp:value:0(normalization_9/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_9/Reshape_1?
normalization_9/subSub'concatenate_18/PartitionedCall:output:0 normalization_9/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization_9/sub?
normalization_9/SqrtSqrt"normalization_9/Reshape_1:output:0*
T0*
_output_shapes

:2
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

:2
normalization_9/Maximum?
normalization_9/truedivRealDivnormalization_9/sub:z:0normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????2
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
category_encoding_9/Minz
category_encoding_9/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :R2
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
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=822"
 category_encoding_9/Assert/Const?
(category_encoding_9/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=822*
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
dtype0	*
value	B	 RR2(
&category_encoding_9/bincount/minlength?
$category_encoding_9/bincount/MaximumMaximum/category_encoding_9/bincount/minlength:output:0$category_encoding_9/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_9/bincount/Maximum?
&category_encoding_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 RR2(
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

Tidx0	*'
_output_shapes
:?????????R*
binary_output(2,
*category_encoding_9/bincount/DenseBincount?
concatenate_19/PartitionedCallPartitionedCallnormalization_9/truediv:z:03category_encoding_9/bincount/DenseBincount:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????W* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_19_layer_call_and_return_conditional_losses_357452 
concatenate_19/PartitionedCall?
IdentityIdentity'concatenate_19/PartitionedCall:output:0"^category_encoding_9/Assert/Assert'^normalization_9/Reshape/ReadVariableOp)^normalization_9/Reshape_1/ReadVariableOp9^string_lookup_9/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????W2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes|
z:?????????:?????????:?????????:?????????:?????????:?????????: : : : 2F
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
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
,__inference_sequential_9_layer_call_fn_37014

inputs
unknown:	W?
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_360792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????W: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????W
 
_user_specified_nameinputs
?	
?
,__inference_sequential_9_layer_call_fn_36225
dense_36_input
unknown:	W?
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_361852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????W: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????W
(
_user_specified_namedense_36_input
?
?
(__inference_dense_37_layer_call_fn_37215

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
C__inference_dense_37_layer_call_and_return_conditional_losses_360402
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
(__inference_model_18_layer_call_fn_35759
area
garden_area

month_sold	
rooms
square_meter
	year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallareagarden_area
month_soldroomssquare_meter	year_soldunknown	unknown_0	unknown_1	unknown_2*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????W*$
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_18_layer_call_and_return_conditional_losses_357482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????W2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes|
z:?????????:?????????:?????????:?????????:?????????:?????????: : : : 22
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
_user_specified_namesquare_meter:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:

_output_shapes
: 
?	
?
C__inference_dense_38_layer_call_and_return_conditional_losses_36056

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
?
?
(__inference_dense_39_layer_call_fn_37253

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
C__inference_dense_39_layer_call_and_return_conditional_losses_360722
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
?
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_36273
dense_36_input!
dense_36_36252:	W?
dense_36_36254:	?!
dense_37_36257:	?@
dense_37_36259:@ 
dense_38_36262:@ 
dense_38_36264:  
dense_39_36267: 
dense_39_36269:
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall? dense_38/StatefulPartitionedCall? dense_39/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCalldense_36_inputdense_36_36252dense_36_36254*
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
C__inference_dense_36_layer_call_and_return_conditional_losses_360242"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_36257dense_37_36259*
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
C__inference_dense_37_layer_call_and_return_conditional_losses_360402"
 dense_37/StatefulPartitionedCall?
 dense_38/StatefulPartitionedCallStatefulPartitionedCall)dense_37/StatefulPartitionedCall:output:0dense_38_36262dense_38_36264*
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
C__inference_dense_38_layer_call_and_return_conditional_losses_360562"
 dense_38/StatefulPartitionedCall?
 dense_39/StatefulPartitionedCallStatefulPartitionedCall)dense_38/StatefulPartitionedCall:output:0dense_39_36267dense_39_36269*
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
C__inference_dense_39_layer_call_and_return_conditional_losses_360722"
 dense_39/StatefulPartitionedCall?
IdentityIdentity)dense_39/StatefulPartitionedCall:output:0!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????W: : : : : : : : 2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall:W S
'
_output_shapes
:?????????W
(
_user_specified_namedense_36_input
?
?
C__inference_dense_36_layer_call_and_return_conditional_losses_36024

inputs1
matmul_readvariableop_resource:	W?.
biasadd_readvariableop_resource:	?

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	W?*
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
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidc
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
mul\
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:??????????2

Identity?
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-36017*<
_output_shapes*
(:??????????:??????????2
	IdentityN?

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????W: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????W
 
_user_specified_nameinputs
А
?
C__inference_model_19_layer_call_and_return_conditional_losses_36839
inputs_area
inputs_garden_area
inputs_month_sold
inputs_rooms
inputs_square_meter
inputs_year_soldR
Nmodel_18_string_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleS
Omodel_18_string_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value	F
8model_18_normalization_9_reshape_readvariableop_resource:H
:model_18_normalization_9_reshape_1_readvariableop_resource:G
4sequential_9_dense_36_matmul_readvariableop_resource:	W?D
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
model_18/concatenate_18/concatConcatV2inputs_roomsinputs_square_meterinputs_garden_areainputs_year_soldinputs_month_sold,model_18/concatenate_18/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2 
model_18/concatenate_18/concat?
/model_18/normalization_9/Reshape/ReadVariableOpReadVariableOp8model_18_normalization_9_reshape_readvariableop_resource*
_output_shapes
:*
dtype021
/model_18/normalization_9/Reshape/ReadVariableOp?
&model_18/normalization_9/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2(
&model_18/normalization_9/Reshape/shape?
 model_18/normalization_9/ReshapeReshape7model_18/normalization_9/Reshape/ReadVariableOp:value:0/model_18/normalization_9/Reshape/shape:output:0*
T0*
_output_shapes

:2"
 model_18/normalization_9/Reshape?
1model_18/normalization_9/Reshape_1/ReadVariableOpReadVariableOp:model_18_normalization_9_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype023
1model_18/normalization_9/Reshape_1/ReadVariableOp?
(model_18/normalization_9/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2*
(model_18/normalization_9/Reshape_1/shape?
"model_18/normalization_9/Reshape_1Reshape9model_18/normalization_9/Reshape_1/ReadVariableOp:value:01model_18/normalization_9/Reshape_1/shape:output:0*
T0*
_output_shapes

:2$
"model_18/normalization_9/Reshape_1?
model_18/normalization_9/subSub'model_18/concatenate_18/concat:output:0)model_18/normalization_9/Reshape:output:0*
T0*'
_output_shapes
:?????????2
model_18/normalization_9/sub?
model_18/normalization_9/SqrtSqrt+model_18/normalization_9/Reshape_1:output:0*
T0*
_output_shapes

:2
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

:2"
 model_18/normalization_9/Maximum?
 model_18/normalization_9/truedivRealDiv model_18/normalization_9/sub:z:0$model_18/normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????2"
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
dtype0*
value	B :R2%
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
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=822+
)model_18/category_encoding_9/Assert/Const?
1model_18/category_encoding_9/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=8223
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
dtype0	*
value	B	 RR21
/model_18/category_encoding_9/bincount/minlength?
-model_18/category_encoding_9/bincount/MaximumMaximum8model_18/category_encoding_9/bincount/minlength:output:0-model_18/category_encoding_9/bincount/mul:z:0*
T0	*
_output_shapes
: 2/
-model_18/category_encoding_9/bincount/Maximum?
/model_18/category_encoding_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 RR21
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

Tidx0	*'
_output_shapes
:?????????R*
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
T0*'
_output_shapes
:?????????W2 
model_18/concatenate_19/concat?
+sequential_9/dense_36/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_36_matmul_readvariableop_resource*
_output_shapes
:	W?*
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
sequential_9/dense_36/SigmoidSigmoid&sequential_9/dense_36/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_36/Sigmoid?
sequential_9/dense_36/mulMul&sequential_9/dense_36/BiasAdd:output:0!sequential_9/dense_36/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
sequential_9/dense_36/mul?
sequential_9/dense_36/IdentityIdentitysequential_9/dense_36/mul:z:0*
T0*(
_output_shapes
:??????????2 
sequential_9/dense_36/Identity?
sequential_9/dense_36/IdentityN	IdentityNsequential_9/dense_36/mul:z:0&sequential_9/dense_36/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-36814*<
_output_shapes*
(:??????????:??????????2!
sequential_9/dense_36/IdentityN?
+sequential_9/dense_37/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_37_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02-
+sequential_9/dense_37/MatMul/ReadVariableOp?
sequential_9/dense_37/MatMulMatMul(sequential_9/dense_36/IdentityN:output:03sequential_9/dense_37/MatMul/ReadVariableOp:value:0*
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
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2X
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
_user_specified_nameinputs/square_meter:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/year_sold:

_output_shapes
: 
?
?
C__inference_model_19_layer_call_and_return_conditional_losses_36420

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
model_18_36393
model_18_36395	
model_18_36397:
model_18_36399:%
sequential_9_36402:	W?!
sequential_9_36404:	?%
sequential_9_36406:	?@ 
sequential_9_36408:@$
sequential_9_36410:@  
sequential_9_36412: $
sequential_9_36414:  
sequential_9_36416:
identity?? model_18/StatefulPartitionedCall?$sequential_9/StatefulPartitionedCall?
 model_18/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5model_18_36393model_18_36395model_18_36397model_18_36399*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????W*$
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_18_layer_call_and_return_conditional_losses_358582"
 model_18/StatefulPartitionedCall?
$sequential_9/StatefulPartitionedCallStatefulPartitionedCall)model_18/StatefulPartitionedCall:output:0sequential_9_36402sequential_9_36404sequential_9_36406sequential_9_36408sequential_9_36410sequential_9_36412sequential_9_36414sequential_9_36416*
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_361852&
$sequential_9/StatefulPartitionedCall?
IdentityIdentity-sequential_9/StatefulPartitionedCall:output:0!^model_18/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2D
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
 
_user_specified_nameinputs:

_output_shapes
: 
?)
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_37069

inputs:
'dense_36_matmul_readvariableop_resource:	W?7
(dense_36_biasadd_readvariableop_resource:	?:
'dense_37_matmul_readvariableop_resource:	?@6
(dense_37_biasadd_readvariableop_resource:@9
'dense_38_matmul_readvariableop_resource:@ 6
(dense_38_biasadd_readvariableop_resource: 9
'dense_39_matmul_readvariableop_resource: 6
(dense_39_biasadd_readvariableop_resource:
identity??dense_36/BiasAdd/ReadVariableOp?dense_36/MatMul/ReadVariableOp?dense_37/BiasAdd/ReadVariableOp?dense_37/MatMul/ReadVariableOp?dense_38/BiasAdd/ReadVariableOp?dense_38/MatMul/ReadVariableOp?dense_39/BiasAdd/ReadVariableOp?dense_39/MatMul/ReadVariableOp?
dense_36/MatMul/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource*
_output_shapes
:	W?*
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
dense_36/BiasAdd}
dense_36/SigmoidSigmoiddense_36/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_36/Sigmoid?
dense_36/mulMuldense_36/BiasAdd:output:0dense_36/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
dense_36/mulw
dense_36/IdentityIdentitydense_36/mul:z:0*
T0*(
_output_shapes
:??????????2
dense_36/Identity?
dense_36/IdentityN	IdentityNdense_36/mul:z:0dense_36/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-37044*<
_output_shapes*
(:??????????:??????????2
dense_36/IdentityN?
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02 
dense_37/MatMul/ReadVariableOp?
dense_37/MatMulMatMuldense_36/IdentityN:output:0&dense_37/MatMul/ReadVariableOp:value:0*
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
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????W: : : : : : : : 2B
dense_36/BiasAdd/ReadVariableOpdense_36/BiasAdd/ReadVariableOp2@
dense_36/MatMul/ReadVariableOpdense_36/MatMul/ReadVariableOp2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2@
dense_37/MatMul/ReadVariableOpdense_37/MatMul/ReadVariableOp2B
dense_38/BiasAdd/ReadVariableOpdense_38/BiasAdd/ReadVariableOp2@
dense_38/MatMul/ReadVariableOpdense_38/MatMul/ReadVariableOp2B
dense_39/BiasAdd/ReadVariableOpdense_39/BiasAdd/ReadVariableOp2@
dense_39/MatMul/ReadVariableOpdense_39/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????W
 
_user_specified_nameinputs
?	
?
C__inference_dense_37_layer_call_and_return_conditional_losses_37225

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
?
?
C__inference_model_19_layer_call_and_return_conditional_losses_36317

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
model_18_36290
model_18_36292	
model_18_36294:
model_18_36296:%
sequential_9_36299:	W?!
sequential_9_36301:	?%
sequential_9_36303:	?@ 
sequential_9_36305:@$
sequential_9_36307:@  
sequential_9_36309: $
sequential_9_36311:  
sequential_9_36313:
identity?? model_18/StatefulPartitionedCall?$sequential_9/StatefulPartitionedCall?
 model_18/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5model_18_36290model_18_36292model_18_36294model_18_36296*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????W*$
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_18_layer_call_and_return_conditional_losses_357482"
 model_18/StatefulPartitionedCall?
$sequential_9/StatefulPartitionedCallStatefulPartitionedCall)model_18/StatefulPartitionedCall:output:0sequential_9_36299sequential_9_36301sequential_9_36303sequential_9_36305sequential_9_36307sequential_9_36309sequential_9_36311sequential_9_36313*
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_360792&
$sequential_9/StatefulPartitionedCall?
IdentityIdentity-sequential_9/StatefulPartitionedCall:output:0!^model_18/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2D
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
 
_user_specified_nameinputs:

_output_shapes
: 
??
?
!__inference__traced_restore_37562
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: 5
"assignvariableop_5_dense_36_kernel:	W?/
 assignvariableop_6_dense_36_bias:	?5
"assignvariableop_7_dense_37_kernel:	?@.
 assignvariableop_8_dense_37_bias:@4
"assignvariableop_9_dense_38_kernel:@ /
!assignvariableop_10_dense_38_bias: 5
#assignvariableop_11_dense_39_kernel: /
!assignvariableop_12_dense_39_bias:&
assignvariableop_13_mean:*
assignvariableop_14_variance:#
assignvariableop_15_count:	 c
Ystring_lookup_9_index_table_table_restore_lookuptableimportv2_string_lookup_9_index_table: #
assignvariableop_16_total: %
assignvariableop_17_count_1: =
*assignvariableop_18_adam_dense_36_kernel_m:	W?7
(assignvariableop_19_adam_dense_36_bias_m:	?=
*assignvariableop_20_adam_dense_37_kernel_m:	?@6
(assignvariableop_21_adam_dense_37_bias_m:@<
*assignvariableop_22_adam_dense_38_kernel_m:@ 6
(assignvariableop_23_adam_dense_38_bias_m: <
*assignvariableop_24_adam_dense_39_kernel_m: 6
(assignvariableop_25_adam_dense_39_bias_m:=
*assignvariableop_26_adam_dense_36_kernel_v:	W?7
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
?
?
C__inference_model_19_layer_call_and_return_conditional_losses_36551
area
garden_area

month_sold	
rooms
square_meter
	year_sold
model_18_36524
model_18_36526	
model_18_36528:
model_18_36530:%
sequential_9_36533:	W?!
sequential_9_36535:	?%
sequential_9_36537:	?@ 
sequential_9_36539:@$
sequential_9_36541:@  
sequential_9_36543: $
sequential_9_36545:  
sequential_9_36547:
identity?? model_18/StatefulPartitionedCall?$sequential_9/StatefulPartitionedCall?
 model_18/StatefulPartitionedCallStatefulPartitionedCallareagarden_area
month_soldroomssquare_meter	year_soldmodel_18_36524model_18_36526model_18_36528model_18_36530*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????W*$
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_18_layer_call_and_return_conditional_losses_358582"
 model_18/StatefulPartitionedCall?
$sequential_9/StatefulPartitionedCallStatefulPartitionedCall)model_18/StatefulPartitionedCall:output:0sequential_9_36533sequential_9_36535sequential_9_36537sequential_9_36539sequential_9_36541sequential_9_36543sequential_9_36545sequential_9_36547*
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_361852&
$sequential_9/StatefulPartitionedCall?
IdentityIdentity-sequential_9/StatefulPartitionedCall:output:0!^model_18/StatefulPartitionedCall%^sequential_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2D
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
_user_specified_namesquare_meter:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:

_output_shapes
: 
?
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_36185

inputs!
dense_36_36164:	W?
dense_36_36166:	?!
dense_37_36169:	?@
dense_37_36171:@ 
dense_38_36174:@ 
dense_38_36176:  
dense_39_36179: 
dense_39_36181:
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall? dense_38/StatefulPartitionedCall? dense_39/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCallinputsdense_36_36164dense_36_36166*
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
C__inference_dense_36_layer_call_and_return_conditional_losses_360242"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_36169dense_37_36171*
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
C__inference_dense_37_layer_call_and_return_conditional_losses_360402"
 dense_37/StatefulPartitionedCall?
 dense_38/StatefulPartitionedCallStatefulPartitionedCall)dense_37/StatefulPartitionedCall:output:0dense_38_36174dense_38_36176*
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
C__inference_dense_38_layer_call_and_return_conditional_losses_360562"
 dense_38/StatefulPartitionedCall?
 dense_39/StatefulPartitionedCallStatefulPartitionedCall)dense_38/StatefulPartitionedCall:output:0dense_39_36179dense_39_36181*
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
C__inference_dense_39_layer_call_and_return_conditional_losses_360722"
 dense_39/StatefulPartitionedCall?
IdentityIdentity)dense_39/StatefulPartitionedCall:output:0!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????W: : : : : : : : 2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall:O K
'
_output_shapes
:?????????W
 
_user_specified_nameinputs
?
?
I__inference_concatenate_18_layer_call_and_return_conditional_losses_35693

inputs
inputs_1
inputs_2
inputs_3
inputs_4
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:?????????:?????????:?????????:?????????:?????????:O K
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
 
_user_specified_nameinputs
?T
?
C__inference_model_18_layer_call_and_return_conditional_losses_36001
area
garden_area

month_sold	
rooms
square_meter
	year_soldI
Estring_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value	=
/normalization_9_reshape_readvariableop_resource:?
1normalization_9_reshape_1_readvariableop_resource:
identity??!category_encoding_9/Assert/Assert?&normalization_9/Reshape/ReadVariableOp?(normalization_9/Reshape_1/ReadVariableOp?8string_lookup_9/None_lookup_table_find/LookupTableFindV2?
8string_lookup_9/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_9_none_lookup_table_find_lookuptablefindv2_table_handleareaFstring_lookup_9_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_9/None_lookup_table_find/LookupTableFindV2?
concatenate_18/PartitionedCallPartitionedCallroomssquare_metergarden_area	year_sold
month_sold*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_18_layer_call_and_return_conditional_losses_356932 
concatenate_18/PartitionedCall?
&normalization_9/Reshape/ReadVariableOpReadVariableOp/normalization_9_reshape_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization_9/Reshape/ReadVariableOp?
normalization_9/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_9/Reshape/shape?
normalization_9/ReshapeReshape.normalization_9/Reshape/ReadVariableOp:value:0&normalization_9/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_9/Reshape?
(normalization_9/Reshape_1/ReadVariableOpReadVariableOp1normalization_9_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02*
(normalization_9/Reshape_1/ReadVariableOp?
normalization_9/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_9/Reshape_1/shape?
normalization_9/Reshape_1Reshape0normalization_9/Reshape_1/ReadVariableOp:value:0(normalization_9/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_9/Reshape_1?
normalization_9/subSub'concatenate_18/PartitionedCall:output:0 normalization_9/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization_9/sub?
normalization_9/SqrtSqrt"normalization_9/Reshape_1:output:0*
T0*
_output_shapes

:2
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

:2
normalization_9/Maximum?
normalization_9/truedivRealDivnormalization_9/sub:z:0normalization_9/Maximum:z:0*
T0*'
_output_shapes
:?????????2
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
category_encoding_9/Minz
category_encoding_9/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :R2
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
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=822"
 category_encoding_9/Assert/Const?
(category_encoding_9/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=822*
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
dtype0	*
value	B	 RR2(
&category_encoding_9/bincount/minlength?
$category_encoding_9/bincount/MaximumMaximum/category_encoding_9/bincount/minlength:output:0$category_encoding_9/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_9/bincount/Maximum?
&category_encoding_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 RR2(
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

Tidx0	*'
_output_shapes
:?????????R*
binary_output(2,
*category_encoding_9/bincount/DenseBincount?
concatenate_19/PartitionedCallPartitionedCallnormalization_9/truediv:z:03category_encoding_9/bincount/DenseBincount:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????W* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_19_layer_call_and_return_conditional_losses_357452 
concatenate_19/PartitionedCall?
IdentityIdentity'concatenate_19/PartitionedCall:output:0"^category_encoding_9/Assert/Assert'^normalization_9/Reshape/ReadVariableOp)^normalization_9/Reshape_1/ReadVariableOp9^string_lookup_9/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????W2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes|
z:?????????:?????????:?????????:?????????:?????????:?????????: : : : 2F
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
_user_specified_namesquare_meter:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:

_output_shapes
: 
?
s
I__inference_concatenate_19_layer_call_and_return_conditional_losses_35745

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????W2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????W2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????R:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????R
 
_user_specified_nameinputs
?)
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_37103

inputs:
'dense_36_matmul_readvariableop_resource:	W?7
(dense_36_biasadd_readvariableop_resource:	?:
'dense_37_matmul_readvariableop_resource:	?@6
(dense_37_biasadd_readvariableop_resource:@9
'dense_38_matmul_readvariableop_resource:@ 6
(dense_38_biasadd_readvariableop_resource: 9
'dense_39_matmul_readvariableop_resource: 6
(dense_39_biasadd_readvariableop_resource:
identity??dense_36/BiasAdd/ReadVariableOp?dense_36/MatMul/ReadVariableOp?dense_37/BiasAdd/ReadVariableOp?dense_37/MatMul/ReadVariableOp?dense_38/BiasAdd/ReadVariableOp?dense_38/MatMul/ReadVariableOp?dense_39/BiasAdd/ReadVariableOp?dense_39/MatMul/ReadVariableOp?
dense_36/MatMul/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource*
_output_shapes
:	W?*
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
dense_36/BiasAdd}
dense_36/SigmoidSigmoiddense_36/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_36/Sigmoid?
dense_36/mulMuldense_36/BiasAdd:output:0dense_36/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
dense_36/mulw
dense_36/IdentityIdentitydense_36/mul:z:0*
T0*(
_output_shapes
:??????????2
dense_36/Identity?
dense_36/IdentityN	IdentityNdense_36/mul:z:0dense_36/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-37078*<
_output_shapes*
(:??????????:??????????2
dense_36/IdentityN?
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02 
dense_37/MatMul/ReadVariableOp?
dense_37/MatMulMatMuldense_36/IdentityN:output:0&dense_37/MatMul/ReadVariableOp:value:0*
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
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????W: : : : : : : : 2B
dense_36/BiasAdd/ReadVariableOpdense_36/BiasAdd/ReadVariableOp2@
dense_36/MatMul/ReadVariableOpdense_36/MatMul/ReadVariableOp2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2@
dense_37/MatMul/ReadVariableOpdense_37/MatMul/ReadVariableOp2B
dense_38/BiasAdd/ReadVariableOpdense_38/BiasAdd/ReadVariableOp2@
dense_38/MatMul/ReadVariableOpdense_38/MatMul/ReadVariableOp2B
dense_39/BiasAdd/ReadVariableOpdense_39/BiasAdd/ReadVariableOp2@
dense_39/MatMul/ReadVariableOpdense_39/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????W
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
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
layer_with_weights-0
layer-6
layer_with_weights-1
layer-7
		optimizer

regularization_losses
trainable_variables
	variables
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"??
_tf_keras_network??{"name": "model_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_19", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}, "name": "area", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "garden_area"}, "name": "garden_area", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}, "name": "month_sold", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}, "name": "rooms", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}, "name": "square_meter", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}, "name": "year_sold", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "Functional", "config": {"name": "model_18", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}, "name": "rooms", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}, "name": "square_meter", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "garden_area"}, "name": "garden_area", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}, "name": "year_sold", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}, "name": "month_sold", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}, "name": "area", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Concatenate", "config": {"name": "concatenate_18", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_18", "inbound_nodes": [[["rooms", 0, 0, {}], ["square_meter", 0, 0, {}], ["garden_area", 0, 0, {}], ["year_sold", 0, 0, {}], ["month_sold", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_9", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "output_mode": "int", "pad_to_max_tokens": false, "vocabulary_size": 82, "vocabulary": null, "encoding": "utf-8"}, "name": "string_lookup_9", "inbound_nodes": [[["area", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "name": "normalization_9", "inbound_nodes": [[["concatenate_18", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_9", "trainable": true, "dtype": "float32", "num_tokens": 82, "output_mode": "binary", "sparse": false}, "name": "category_encoding_9", "inbound_nodes": [[["string_lookup_9", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_19", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_19", "inbound_nodes": [[["normalization_9", 0, 0, {}], ["category_encoding_9", 0, 0, {}]]]}], "input_layers": {"area": ["area", 0, 0], "rooms": ["rooms", 0, 0], "square_meter": ["square_meter", 0, 0], "garden_area": ["garden_area", 0, 0], "year_sold": ["year_sold", 0, 0], "month_sold": ["month_sold", 0, 0]}, "output_layers": [["concatenate_19", 0, 0]]}, "name": "model_18", "inbound_nodes": [{"area": ["area", 0, 0, {}], "rooms": ["rooms", 0, 0, {}], "square_meter": ["square_meter", 0, 0, {}], "garden_area": ["garden_area", 0, 0, {}], "year_sold": ["year_sold", 0, 0, {}], "month_sold": ["month_sold", 0, 0, {}]}]}, {"class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 87]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_36_input"}}, {"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 128, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential_9", "inbound_nodes": [[["model_18", 1, 0, {}]]]}], "input_layers": {"area": ["area", 0, 0], "rooms": ["rooms", 0, 0], "square_meter": ["square_meter", 0, 0], "garden_area": ["garden_area", 0, 0], "year_sold": ["year_sold", 0, 0], "month_sold": ["month_sold", 0, 0]}, "output_layers": [["sequential_9", 1, 0]]}, "shared_object_id": 26, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"area": {"class_name": "TensorShape", "items": [null, 1]}, "rooms": {"class_name": "TensorShape", "items": [null, 1]}, "square_meter": {"class_name": "TensorShape", "items": [null, 1]}, "garden_area": {"class_name": "TensorShape", "items": [null, 1]}, "year_sold": {"class_name": "TensorShape", "items": [null, 1]}, "month_sold": {"class_name": "TensorShape", "items": [null, 1]}}, "is_graph_network": true, "save_spec": {"area": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "string", "area"]}, "rooms": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "rooms"]}, "square_meter": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "square_meter"]}, "garden_area": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "garden_area"]}, "year_sold": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "year_sold"]}, "month_sold": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "month_sold"]}}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_19", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}, "name": "area", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "garden_area"}, "name": "garden_area", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}, "name": "month_sold", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}, "name": "rooms", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}, "name": "square_meter", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}, "name": "year_sold", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "Functional", "config": {"name": "model_18", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}, "name": "rooms", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}, "name": "square_meter", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "garden_area"}, "name": "garden_area", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}, "name": "year_sold", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}, "name": "month_sold", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}, "name": "area", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Concatenate", "config": {"name": "concatenate_18", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_18", "inbound_nodes": [[["rooms", 0, 0, {}], ["square_meter", 0, 0, {}], ["garden_area", 0, 0, {}], ["year_sold", 0, 0, {}], ["month_sold", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_9", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "output_mode": "int", "pad_to_max_tokens": false, "vocabulary_size": 82, "vocabulary": null, "encoding": "utf-8"}, "name": "string_lookup_9", "inbound_nodes": [[["area", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "name": "normalization_9", "inbound_nodes": [[["concatenate_18", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_9", "trainable": true, "dtype": "float32", "num_tokens": 82, "output_mode": "binary", "sparse": false}, "name": "category_encoding_9", "inbound_nodes": [[["string_lookup_9", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_19", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_19", "inbound_nodes": [[["normalization_9", 0, 0, {}], ["category_encoding_9", 0, 0, {}]]]}], "input_layers": {"area": ["area", 0, 0], "rooms": ["rooms", 0, 0], "square_meter": ["square_meter", 0, 0], "garden_area": ["garden_area", 0, 0], "year_sold": ["year_sold", 0, 0], "month_sold": ["month_sold", 0, 0]}, "output_layers": [["concatenate_19", 0, 0]]}, "name": "model_18", "inbound_nodes": [{"area": ["area", 0, 0, {}], "rooms": ["rooms", 0, 0, {}], "square_meter": ["square_meter", 0, 0, {}], "garden_area": ["garden_area", 0, 0, {}], "year_sold": ["year_sold", 0, 0, {}], "month_sold": ["month_sold", 0, 0, {}]}], "shared_object_id": 11}, {"class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 87]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_36_input"}}, {"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 128, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential_9", "inbound_nodes": [[["model_18", 1, 0, {}]]], "shared_object_id": 25}], "input_layers": {"area": ["area", 0, 0], "rooms": ["rooms", 0, 0], "square_meter": ["square_meter", 0, 0], "garden_area": ["garden_area", 0, 0], "year_sold": ["year_sold", 0, 0], "month_sold": ["month_sold", 0, 0]}, "output_layers": [["sequential_9", 1, 0]]}}, "training_config": {"loss": {"class_name": "MeanAbsoluteError", "config": {"reduction": "auto", "name": "mean_absolute_error"}, "shared_object_id": 33}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
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
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "year_sold", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}}
?N
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-0
layer-7
layer_with_weights-1
layer-8
layer-9
layer-10
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?L
_tf_keras_network?K{"name": "model_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_18", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}, "name": "rooms", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}, "name": "square_meter", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "garden_area"}, "name": "garden_area", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}, "name": "year_sold", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}, "name": "month_sold", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}, "name": "area", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Concatenate", "config": {"name": "concatenate_18", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_18", "inbound_nodes": [[["rooms", 0, 0, {}], ["square_meter", 0, 0, {}], ["garden_area", 0, 0, {}], ["year_sold", 0, 0, {}], ["month_sold", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_9", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "output_mode": "int", "pad_to_max_tokens": false, "vocabulary_size": 82, "vocabulary": null, "encoding": "utf-8"}, "name": "string_lookup_9", "inbound_nodes": [[["area", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "name": "normalization_9", "inbound_nodes": [[["concatenate_18", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_9", "trainable": true, "dtype": "float32", "num_tokens": 82, "output_mode": "binary", "sparse": false}, "name": "category_encoding_9", "inbound_nodes": [[["string_lookup_9", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_19", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_19", "inbound_nodes": [[["normalization_9", 0, 0, {}], ["category_encoding_9", 0, 0, {}]]]}], "input_layers": {"area": ["area", 0, 0], "rooms": ["rooms", 0, 0], "square_meter": ["square_meter", 0, 0], "garden_area": ["garden_area", 0, 0], "year_sold": ["year_sold", 0, 0], "month_sold": ["month_sold", 0, 0]}, "output_layers": [["concatenate_19", 0, 0]]}, "inbound_nodes": [{"area": ["area", 0, 0, {}], "rooms": ["rooms", 0, 0, {}], "square_meter": ["square_meter", 0, 0, {}], "garden_area": ["garden_area", 0, 0, {}], "year_sold": ["year_sold", 0, 0, {}], "month_sold": ["month_sold", 0, 0, {}]}], "shared_object_id": 11, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"area": {"class_name": "TensorShape", "items": [null, 1]}, "rooms": {"class_name": "TensorShape", "items": [null, 1]}, "square_meter": {"class_name": "TensorShape", "items": [null, 1]}, "garden_area": {"class_name": "TensorShape", "items": [null, 1]}, "year_sold": {"class_name": "TensorShape", "items": [null, 1]}, "month_sold": {"class_name": "TensorShape", "items": [null, 1]}}, "is_graph_network": true, "save_spec": {"area": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "string", "area"]}, "rooms": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "rooms"]}, "square_meter": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "square_meter"]}, "garden_area": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "garden_area"]}, "year_sold": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "year_sold"]}, "month_sold": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "month_sold"]}}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_18", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}, "name": "rooms", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}, "name": "square_meter", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "garden_area"}, "name": "garden_area", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}, "name": "year_sold", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}, "name": "month_sold", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}, "name": "area", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Concatenate", "config": {"name": "concatenate_18", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_18", "inbound_nodes": [[["rooms", 0, 0, {}], ["square_meter", 0, 0, {}], ["garden_area", 0, 0, {}], ["year_sold", 0, 0, {}], ["month_sold", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "StringLookup", "config": {"name": "string_lookup_9", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "output_mode": "int", "pad_to_max_tokens": false, "vocabulary_size": 82, "vocabulary": null, "encoding": "utf-8"}, "name": "string_lookup_9", "inbound_nodes": [[["area", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Normalization", "config": {"name": "normalization_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "name": "normalization_9", "inbound_nodes": [[["concatenate_18", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_9", "trainable": true, "dtype": "float32", "num_tokens": 82, "output_mode": "binary", "sparse": false}, "name": "category_encoding_9", "inbound_nodes": [[["string_lookup_9", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Concatenate", "config": {"name": "concatenate_19", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_19", "inbound_nodes": [[["normalization_9", 0, 0, {}], ["category_encoding_9", 0, 0, {}]]], "shared_object_id": 10}], "input_layers": {"area": ["area", 0, 0], "rooms": ["rooms", 0, 0], "square_meter": ["square_meter", 0, 0], "garden_area": ["garden_area", 0, 0], "year_sold": ["year_sold", 0, 0], "month_sold": ["month_sold", 0, 0]}, "output_layers": [["concatenate_19", 0, 0]]}}}
?)
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?'
_tf_keras_sequential?'{"name": "sequential_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 87]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_36_input"}}, {"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 128, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "inbound_nodes": [[["model_18", 1, 0, {}]]], "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 87}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 87]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 87]}, "float32", "dense_36_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 87]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_36_input"}, "shared_object_id": 12}, {"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 128, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18}, {"class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 21}, {"class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24}]}}}
?
 iter

!beta_1

"beta_2
	#decay
$learning_rate%m?&m?'m?(m?)m?*m?+m?,m?%v?&v?'v?(v?)v?*v?+v?,v?"
	optimizer
 "
trackable_list_wrapper
X
%0
&1
'2
(3
)4
*5
+6
,7"
trackable_list_wrapper
o
-1
.2
/3
%4
&5
'6
(7
)8
*9
+10
,11"
trackable_list_wrapper
?
0non_trainable_variables
1layer_metrics

regularization_losses

2layers
3metrics
4layer_regularization_losses
trainable_variables
	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
5regularization_losses
6trainable_variables
7	variables
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "concatenate_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate_18", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["rooms", 0, 0, {}], ["square_meter", 0, 0, {}], ["garden_area", 0, 0, {}], ["year_sold", 0, 0, {}], ["month_sold", 0, 0, {}]]], "shared_object_id": 6, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}]}
?
9state_variables

:_table
;	keras_api"?
_tf_keras_layer?{"name": "string_lookup_9", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": null, "stateful": true, "must_restore_from_config": true, "class_name": "StringLookup", "config": {"name": "string_lookup_9", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "output_mode": "int", "pad_to_max_tokens": false, "vocabulary_size": 82, "vocabulary": null, "encoding": "utf-8"}, "inbound_nodes": [[["area", 0, 0, {}]]], "shared_object_id": 7}
?
<
_keep_axis
=_reduce_axis
>_reduce_axis_mask
?_broadcast_shape
-mean
.variance
	/count
@	keras_api
?_adapt_function"?
_tf_keras_layer?{"name": "normalization_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": true, "must_restore_from_config": true, "class_name": "Normalization", "config": {"name": "normalization_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "inbound_nodes": [[["concatenate_18", 0, 0, {}]]], "shared_object_id": 8, "build_input_shape": [null, 5]}
?
A	keras_api"?
_tf_keras_layer?{"name": "category_encoding_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "CategoryEncoding", "config": {"name": "category_encoding_9", "trainable": true, "dtype": "float32", "num_tokens": 82, "output_mode": "binary", "sparse": false}, "inbound_nodes": [[["string_lookup_9", 0, 0, {}]]], "shared_object_id": 9}
?
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "concatenate_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate_19", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["normalization_9", 0, 0, {}], ["category_encoding_9", 0, 0, {}]]], "shared_object_id": 10, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 5]}, {"class_name": "TensorShape", "items": [null, 82]}]}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
-1
.2
/3"
trackable_list_wrapper
?
Fnon_trainable_variables
Glayer_metrics
regularization_losses

Hlayers
Imetrics
Jlayer_regularization_losses
trainable_variables
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

%kernel
&bias
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 128, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 87}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 87]}}
?

'kernel
(bias
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 41}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

)kernel
*bias
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 42}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

+kernel
,bias
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 43}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
 "
trackable_list_wrapper
X
%0
&1
'2
(3
)4
*5
+6
,7"
trackable_list_wrapper
X
%0
&1
'2
(3
)4
*5
+6
,7"
trackable_list_wrapper
?
[non_trainable_variables
\layer_metrics
regularization_losses

]layers
^metrics
_layer_regularization_losses
trainable_variables
	variables
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
": 	W?2dense_36/kernel
:?2dense_36/bias
": 	?@2dense_37/kernel
:@2dense_37/bias
!:@ 2dense_38/kernel
: 2dense_38/bias
!: 2dense_39/kernel
:2dense_39/bias
:2mean
:2variance
:	 2count
5
-1
.2
/3"
trackable_list_wrapper
 "
trackable_dict_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
'
`0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
anon_trainable_variables
blayer_metrics
5regularization_losses

clayers
dmetrics
elayer_regularization_losses
6trainable_variables
7	variables
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
fnon_trainable_variables
glayer_metrics
Bregularization_losses

hlayers
imetrics
jlayer_regularization_losses
Ctrainable_variables
D	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5
-1
.2
/3"
trackable_list_wrapper
 "
trackable_dict_wrapper
n
0
1
2
3
4
5
6
7
8
9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
?
knon_trainable_variables
llayer_metrics
Kregularization_losses

mlayers
nmetrics
olayer_regularization_losses
Ltrainable_variables
M	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
?
pnon_trainable_variables
qlayer_metrics
Oregularization_losses

rlayers
smetrics
tlayer_regularization_losses
Ptrainable_variables
Q	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
?
unon_trainable_variables
vlayer_metrics
Sregularization_losses

wlayers
xmetrics
ylayer_regularization_losses
Ttrainable_variables
U	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
?
znon_trainable_variables
{layer_metrics
Wregularization_losses

|layers
}metrics
~layer_regularization_losses
Xtrainable_variables
Y	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 44}
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
/
0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
':%	W?2Adam/dense_36/kernel/m
!:?2Adam/dense_36/bias/m
':%	?@2Adam/dense_37/kernel/m
 :@2Adam/dense_37/bias/m
&:$@ 2Adam/dense_38/kernel/m
 : 2Adam/dense_38/bias/m
&:$ 2Adam/dense_39/kernel/m
 :2Adam/dense_39/bias/m
':%	W?2Adam/dense_36/kernel/v
!:?2Adam/dense_36/bias/v
':%	?@2Adam/dense_37/kernel/v
 :@2Adam/dense_37/bias/v
&:$@ 2Adam/dense_38/kernel/v
 : 2Adam/dense_38/bias/v
&:$ 2Adam/dense_39/kernel/v
 :2Adam/dense_39/bias/v
?2?
 __inference__wrapped_model_35663?
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
0
	year_sold#? 
	year_sold?????????
?2?
(__inference_model_19_layer_call_fn_36344
(__inference_model_19_layer_call_fn_36627
(__inference_model_19_layer_call_fn_36661
(__inference_model_19_layer_call_fn_36481?
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
C__inference_model_19_layer_call_and_return_conditional_losses_36750
C__inference_model_19_layer_call_and_return_conditional_losses_36839
C__inference_model_19_layer_call_and_return_conditional_losses_36516
C__inference_model_19_layer_call_and_return_conditional_losses_36551?
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
(__inference_model_18_layer_call_fn_35759
(__inference_model_18_layer_call_fn_36857
(__inference_model_18_layer_call_fn_36875
(__inference_model_18_layer_call_fn_35887?
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
C__inference_model_18_layer_call_and_return_conditional_losses_36934
C__inference_model_18_layer_call_and_return_conditional_losses_36993
C__inference_model_18_layer_call_and_return_conditional_losses_35944
C__inference_model_18_layer_call_and_return_conditional_losses_36001?
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
,__inference_sequential_9_layer_call_fn_36098
,__inference_sequential_9_layer_call_fn_37014
,__inference_sequential_9_layer_call_fn_37035
,__inference_sequential_9_layer_call_fn_36225?
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_37069
G__inference_sequential_9_layer_call_and_return_conditional_losses_37103
G__inference_sequential_9_layer_call_and_return_conditional_losses_36249
G__inference_sequential_9_layer_call_and_return_conditional_losses_36273?
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
?B?
#__inference_signature_wrapper_36593areagarden_area
month_soldroomssquare_meter	year_sold"?
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
.__inference_concatenate_18_layer_call_fn_37112?
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
I__inference_concatenate_18_layer_call_and_return_conditional_losses_37122?
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
__inference_adapt_step_37168?
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
.__inference_concatenate_19_layer_call_fn_37174?
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
I__inference_concatenate_19_layer_call_and_return_conditional_losses_37181?
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
(__inference_dense_36_layer_call_fn_37190?
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
C__inference_dense_36_layer_call_and_return_conditional_losses_37206?
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
(__inference_dense_37_layer_call_fn_37215?
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
C__inference_dense_37_layer_call_and_return_conditional_losses_37225?
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
(__inference_dense_38_layer_call_fn_37234?
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
C__inference_dense_38_layer_call_and_return_conditional_losses_37244?
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
(__inference_dense_39_layer_call_fn_37253?
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
C__inference_dense_39_layer_call_and_return_conditional_losses_37263?
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
__inference__creator_37268?
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
__inference__initializer_37273?
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
__inference__destroyer_37278?
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
__inference_save_fn_37297checkpoint_key"?
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
__inference_restore_fn_37305restored_tensors_0restored_tensors_1"?
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
__inference__creator_37268?

? 
? "? 8
__inference__destroyer_37278?

? 
? "? :
__inference__initializer_37273?

? 
? "? ?
 __inference__wrapped_model_35663?:?-.%&'()*+,???
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
0
	year_sold#? 
	year_sold?????????
? ";?8
6
sequential_9&?#
sequential_9?????????l
__inference_adapt_step_37168L/-.A?>
7?4
2?/?
??????????IteratorSpec
? "
 ?
I__inference_concatenate_18_layer_call_and_return_conditional_losses_37122????
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
? "%?"
?
0?????????
? ?
.__inference_concatenate_18_layer_call_fn_37112????
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
? "???????????
I__inference_concatenate_19_layer_call_and_return_conditional_losses_37181?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????R
? "%?"
?
0?????????W
? ?
.__inference_concatenate_19_layer_call_fn_37174vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????R
? "??????????W?
C__inference_dense_36_layer_call_and_return_conditional_losses_37206]%&/?,
%?"
 ?
inputs?????????W
? "&?#
?
0??????????
? |
(__inference_dense_36_layer_call_fn_37190P%&/?,
%?"
 ?
inputs?????????W
? "????????????
C__inference_dense_37_layer_call_and_return_conditional_losses_37225]'(0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? |
(__inference_dense_37_layer_call_fn_37215P'(0?-
&?#
!?
inputs??????????
? "??????????@?
C__inference_dense_38_layer_call_and_return_conditional_losses_37244\)*/?,
%?"
 ?
inputs?????????@
? "%?"
?
0????????? 
? {
(__inference_dense_38_layer_call_fn_37234O)*/?,
%?"
 ?
inputs?????????@
? "?????????? ?
C__inference_dense_39_layer_call_and_return_conditional_losses_37263\+,/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? {
(__inference_dense_39_layer_call_fn_37253O+,/?,
%?"
 ?
inputs????????? 
? "???????????
C__inference_model_18_layer_call_and_return_conditional_losses_35944?:?-.???
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
0
	year_sold#? 
	year_sold?????????
p 

 
? "%?"
?
0?????????W
? ?
C__inference_model_18_layer_call_and_return_conditional_losses_36001?:?-.???
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
0
	year_sold#? 
	year_sold?????????
p

 
? "%?"
?
0?????????W
? ?
C__inference_model_18_layer_call_and_return_conditional_losses_36934?:?-.???
???
???
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
7
	year_sold*?'
inputs/year_sold?????????
p 

 
? "%?"
?
0?????????W
? ?
C__inference_model_18_layer_call_and_return_conditional_losses_36993?:?-.???
???
???
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
7
	year_sold*?'
inputs/year_sold?????????
p

 
? "%?"
?
0?????????W
? ?
(__inference_model_18_layer_call_fn_35759?:?-.???
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
0
	year_sold#? 
	year_sold?????????
p 

 
? "??????????W?
(__inference_model_18_layer_call_fn_35887?:?-.???
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
0
	year_sold#? 
	year_sold?????????
p

 
? "??????????W?
(__inference_model_18_layer_call_fn_36857?:?-.???
???
???
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
7
	year_sold*?'
inputs/year_sold?????????
p 

 
? "??????????W?
(__inference_model_18_layer_call_fn_36875?:?-.???
???
???
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
7
	year_sold*?'
inputs/year_sold?????????
p

 
? "??????????W?
C__inference_model_19_layer_call_and_return_conditional_losses_36516?:?-.%&'()*+,???
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
0
	year_sold#? 
	year_sold?????????
p 

 
? "%?"
?
0?????????
? ?
C__inference_model_19_layer_call_and_return_conditional_losses_36551?:?-.%&'()*+,???
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
0
	year_sold#? 
	year_sold?????????
p

 
? "%?"
?
0?????????
? ?
C__inference_model_19_layer_call_and_return_conditional_losses_36750?:?-.%&'()*+,???
???
???
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
7
	year_sold*?'
inputs/year_sold?????????
p 

 
? "%?"
?
0?????????
? ?
C__inference_model_19_layer_call_and_return_conditional_losses_36839?:?-.%&'()*+,???
???
???
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
(__inference_model_19_layer_call_fn_36344?:?-.%&'()*+,???
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
0
	year_sold#? 
	year_sold?????????
p 

 
? "???????????
(__inference_model_19_layer_call_fn_36481?:?-.%&'()*+,???
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
0
	year_sold#? 
	year_sold?????????
p

 
? "???????????
(__inference_model_19_layer_call_fn_36627?:?-.%&'()*+,???
???
???
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
7
	year_sold*?'
inputs/year_sold?????????
p 

 
? "???????????
(__inference_model_19_layer_call_fn_36661?:?-.%&'()*+,???
???
???
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
7
	year_sold*?'
inputs/year_sold?????????
p

 
? "??????????y
__inference_restore_fn_37305Y:K?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_37297?:&?#
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
G__inference_sequential_9_layer_call_and_return_conditional_losses_36249r%&'()*+,??<
5?2
(?%
dense_36_input?????????W
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_9_layer_call_and_return_conditional_losses_36273r%&'()*+,??<
5?2
(?%
dense_36_input?????????W
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_9_layer_call_and_return_conditional_losses_37069j%&'()*+,7?4
-?*
 ?
inputs?????????W
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_9_layer_call_and_return_conditional_losses_37103j%&'()*+,7?4
-?*
 ?
inputs?????????W
p

 
? "%?"
?
0?????????
? ?
,__inference_sequential_9_layer_call_fn_36098e%&'()*+,??<
5?2
(?%
dense_36_input?????????W
p 

 
? "???????????
,__inference_sequential_9_layer_call_fn_36225e%&'()*+,??<
5?2
(?%
dense_36_input?????????W
p

 
? "???????????
,__inference_sequential_9_layer_call_fn_37014]%&'()*+,7?4
-?*
 ?
inputs?????????W
p 

 
? "???????????
,__inference_sequential_9_layer_call_fn_37035]%&'()*+,7?4
-?*
 ?
inputs?????????W
p

 
? "???????????
#__inference_signature_wrapper_36593?:?-.%&'()*+,???
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
0
	year_sold#? 
	year_sold?????????";?8
6
sequential_9&?#
sequential_9?????????