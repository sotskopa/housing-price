??
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
 ?"serve*	2.5.0-rc02v1.12.1-53831-ga8b6d5ff93a8??
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
`
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean
Y
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
:*
dtype0
h
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance
a
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
:*
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
|
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_12/kernel
u
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel* 
_output_shapes
:
??*
dtype0
s
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_12/bias
l
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes	
:?*
dtype0
{
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@* 
shared_namedense_13/kernel
t
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes
:	?@*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:@*
dtype0
z
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ * 
shared_namedense_14/kernel
s
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes

:@ *
dtype0
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
: *
dtype0
z
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_15/kernel
s
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes

: *
dtype0
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes
:*
dtype0
?
string_lookup_3_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_206512*
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
Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_12/kernel/m
?
*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_12/bias/m
z
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*'
shared_nameAdam/dense_13/kernel/m
?
*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m*
_output_shapes
:	?@*
dtype0
?
Adam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_13/bias/m
y
(Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/dense_14/kernel/m
?
*Adam/dense_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/m*
_output_shapes

:@ *
dtype0
?
Adam/dense_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_14/bias/m
y
(Adam/dense_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_15/kernel/m
?
*Adam/dense_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/m*
_output_shapes

: *
dtype0
?
Adam/dense_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_15/bias/m
y
(Adam/dense_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_12/kernel/v
?
*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_12/bias/v
z
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*'
shared_nameAdam/dense_13/kernel/v
?
*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v*
_output_shapes
:	?@*
dtype0
?
Adam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_13/bias/v
y
(Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_nameAdam/dense_14/kernel/v
?
*Adam/dense_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/v*
_output_shapes

:@ *
dtype0
?
Adam/dense_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_14/bias/v
y
(Adam/dense_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_15/kernel/v
?
*Adam/dense_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/v*
_output_shapes

: *
dtype0
?
Adam/dense_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_15/bias/v
y
(Adam/dense_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/v*
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
GPU2*0J 8? *$
fR
__inference_<lambda>_230365

NoOpNoOp^PartitionedCall
?
Jstring_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2string_lookup_3_index_table*
Tkeys0*
Tvalues0	*.
_class$
" loc:@string_lookup_3_index_table*
_output_shapes

::
?;
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*?;
value?;B?; B?;
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-0
	layer-8

layer_with_weights-1

layer-9
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
 
 
 
 
 
 
 
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
layer-8
layer_with_weights-0
layer-9
layer_with_weights-1
layer-10
layer-11
layer-12
	variables
regularization_losses
trainable_variables
	keras_api
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
regularization_losses
 trainable_variables
!	keras_api
?
"iter

#beta_1

$beta_2
	%decay
&learning_rate*m?+m?,m?-m?.m?/m?0m?1m?*v?+v?,v?-v?.v?/v?0v?1v?
O
'1
(2
)3
*4
+5
,6
-7
.8
/9
010
111
 
8
*0
+1
,2
-3
.4
/5
06
17
?
2non_trainable_variables
	variables

3layers
regularization_losses
4metrics
5layer_regularization_losses
trainable_variables
6layer_metrics
 
R
7	variables
8trainable_variables
9regularization_losses
:	keras_api
0
;state_variables

<_table
=	keras_api
?
>
_keep_axis
?_reduce_axis
@_reduce_axis_mask
A_broadcast_shape
'mean
(variance
	)count
B	keras_api

C	keras_api
R
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api

'1
(2
)3
 
 
?
Hnon_trainable_variables
	variables

Ilayers
regularization_losses
Jmetrics
Klayer_regularization_losses
trainable_variables
Llayer_metrics
h

*kernel
+bias
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
h

,kernel
-bias
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
h

.kernel
/bias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
h

0kernel
1bias
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
8
*0
+1
,2
-3
.4
/5
06
17
 
8
*0
+1
,2
-3
.4
/5
06
17
?
]non_trainable_variables
	variables

^layers
regularization_losses
_metrics
`layer_regularization_losses
 trainable_variables
alayer_metrics
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
@>
VARIABLE_VALUEmean&variables/1/.ATTRIBUTES/VARIABLE_VALUE
DB
VARIABLE_VALUEvariance&variables/2/.ATTRIBUTES/VARIABLE_VALUE
A?
VARIABLE_VALUEcount&variables/3/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_12/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_12/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_13/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_13/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_14/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_14/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_15/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_15/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE

'1
(2
)3
F
0
1
2
3
4
5
6
7
	8

9

b0
 
 
 
 
 
?
7	variables
8trainable_variables

clayers
9regularization_losses
dmetrics
elayer_regularization_losses
fnon_trainable_variables
glayer_metrics
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
D	variables
Etrainable_variables

hlayers
Fregularization_losses
imetrics
jlayer_regularization_losses
knon_trainable_variables
llayer_metrics

'1
(2
)3
^
0
1
2
3
4
5
6
7
8
9
10
11
12
 
 
 

*0
+1

*0
+1
 
?
M	variables
Ntrainable_variables

mlayers
Oregularization_losses
nmetrics
olayer_regularization_losses
pnon_trainable_variables
qlayer_metrics

,0
-1

,0
-1
 
?
Q	variables
Rtrainable_variables

rlayers
Sregularization_losses
smetrics
tlayer_regularization_losses
unon_trainable_variables
vlayer_metrics

.0
/1

.0
/1
 
?
U	variables
Vtrainable_variables

wlayers
Wregularization_losses
xmetrics
ylayer_regularization_losses
znon_trainable_variables
{layer_metrics

00
11

00
11
 
?
Y	variables
Ztrainable_variables

|layers
[regularization_losses
}metrics
~layer_regularization_losses
non_trainable_variables
?layer_metrics
 

0
1
2
3
 
 
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
nl
VARIABLE_VALUEAdam/dense_12/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_12/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_13/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_13/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_14/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_14/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_15/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_15/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_12/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_12/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_13/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_13/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_14/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_14/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_15/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_15/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
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
?
serving_default_operating_costPlaceholder*'
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_areaserving_default_garden_areaserving_default_month_soldserving_default_operating_costserving_default_roomsserving_default_square_meter$serving_default_year_of_constructionserving_default_year_soldstring_lookup_3_index_tableConstmeanvariancedense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs


*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_229628
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpmean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOpJstring_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2Lstring_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2:1total/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp(Adam/dense_13/bias/m/Read/ReadVariableOp*Adam/dense_14/kernel/m/Read/ReadVariableOp(Adam/dense_14/bias/m/Read/ReadVariableOp*Adam/dense_15/kernel/m/Read/ReadVariableOp(Adam/dense_15/bias/m/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp(Adam/dense_13/bias/v/Read/ReadVariableOp*Adam/dense_14/kernel/v/Read/ReadVariableOp(Adam/dense_14/bias/v/Read/ReadVariableOp*Adam/dense_15/kernel/v/Read/ReadVariableOp(Adam/dense_15/bias/v/Read/ReadVariableOpConst_1*1
Tin*
(2&			*
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
GPU2*0J 8? *(
f#R!
__inference__traced_save_230504
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratemeanvariancecountdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/biasstring_lookup_3_index_tabletotalcount_1Adam/dense_12/kernel/mAdam/dense_12/bias/mAdam/dense_13/kernel/mAdam/dense_13/bias/mAdam/dense_14/kernel/mAdam/dense_14/bias/mAdam/dense_15/kernel/mAdam/dense_15/bias/mAdam/dense_12/kernel/vAdam/dense_12/bias/vAdam/dense_13/kernel/vAdam/dense_13/bias/vAdam/dense_14/kernel/vAdam/dense_14/bias/vAdam/dense_15/kernel/vAdam/dense_15/bias/v*/
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
GPU2*0J 8? *+
f&R$
"__inference__traced_restore_230619??
?
s
I__inference_concatenate_7_layer_call_and_return_conditional_losses_228748

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
':?????????:??????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?U
?
C__inference_model_6_layer_call_and_return_conditional_losses_228751

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7I
Estring_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value	=
/normalization_3_reshape_readvariableop_resource:?
1normalization_3_reshape_1_readvariableop_resource:
identity??!category_encoding_3/Assert/Assert?&normalization_3/Reshape/ReadVariableOp?(normalization_3/Reshape_1/ReadVariableOp?8string_lookup_3/None_lookup_table_find/LookupTableFindV2?
8string_lookup_3/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleinputsFstring_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_3/None_lookup_table_find/LookupTableFindV2?
concatenate_6/PartitionedCallPartitionedCallinputs_4inputs_5inputs_1inputs_3inputs_6inputs_7inputs_2*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_6_layer_call_and_return_conditional_losses_2286962
concatenate_6/PartitionedCall?
&normalization_3/Reshape/ReadVariableOpReadVariableOp/normalization_3_reshape_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization_3/Reshape/ReadVariableOp?
normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_3/Reshape/shape?
normalization_3/ReshapeReshape.normalization_3/Reshape/ReadVariableOp:value:0&normalization_3/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_3/Reshape?
(normalization_3/Reshape_1/ReadVariableOpReadVariableOp1normalization_3_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02*
(normalization_3/Reshape_1/ReadVariableOp?
normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_3/Reshape_1/shape?
normalization_3/Reshape_1Reshape0normalization_3/Reshape_1/ReadVariableOp:value:0(normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_3/Reshape_1?
normalization_3/subSub&concatenate_6/PartitionedCall:output:0 normalization_3/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization_3/sub?
normalization_3/SqrtSqrt"normalization_3/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_3/Maximum/y?
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_3/Maximum?
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_3/truediv?
category_encoding_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_3/Const?
category_encoding_3/MaxMaxAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0"category_encoding_3/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_3/Max?
category_encoding_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_3/Const_1?
category_encoding_3/MinMinAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0$category_encoding_3/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_3/Min{
category_encoding_3/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding_3/Cast/x?
category_encoding_3/CastCast#category_encoding_3/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_3/Cast?
category_encoding_3/GreaterGreatercategory_encoding_3/Cast:y:0 category_encoding_3/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_3/Greater~
category_encoding_3/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_3/Cast_1/x?
category_encoding_3/Cast_1Cast%category_encoding_3/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_3/Cast_1?
 category_encoding_3/GreaterEqualGreaterEqual category_encoding_3/Min:output:0category_encoding_3/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/GreaterEqual?
category_encoding_3/LogicalAnd
LogicalAndcategory_encoding_3/Greater:z:0$category_encoding_3/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_3/LogicalAnd?
 category_encoding_3/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=3002"
 category_encoding_3/Assert/Const?
(category_encoding_3/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=3002*
(category_encoding_3/Assert/Assert/data_0?
!category_encoding_3/Assert/AssertAssert"category_encoding_3/LogicalAnd:z:01category_encoding_3/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2#
!category_encoding_3/Assert/Assert?
"category_encoding_3/bincount/ShapeShapeAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_3/bincount/Shape?
"category_encoding_3/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_3/bincount/Const?
!category_encoding_3/bincount/ProdProd+category_encoding_3/bincount/Shape:output:0+category_encoding_3/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_3/bincount/Prod?
&category_encoding_3/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_3/bincount/Greater/y?
$category_encoding_3/bincount/GreaterGreater*category_encoding_3/bincount/Prod:output:0/category_encoding_3/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_3/bincount/Greater?
!category_encoding_3/bincount/CastCast(category_encoding_3/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_3/bincount/Cast?
$category_encoding_3/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_3/bincount/Const_1?
 category_encoding_3/bincount/MaxMaxAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_3/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/Max?
"category_encoding_3/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_3/bincount/add/y?
 category_encoding_3/bincount/addAddV2)category_encoding_3/bincount/Max:output:0+category_encoding_3/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/add?
 category_encoding_3/bincount/mulMul%category_encoding_3/bincount/Cast:y:0$category_encoding_3/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/mul?
&category_encoding_3/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_3/bincount/minlength?
$category_encoding_3/bincount/MaximumMaximum/category_encoding_3/bincount/minlength:output:0$category_encoding_3/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_3/bincount/Maximum?
&category_encoding_3/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_3/bincount/maxlength?
$category_encoding_3/bincount/MinimumMinimum/category_encoding_3/bincount/maxlength:output:0(category_encoding_3/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_3/bincount/Minimum?
$category_encoding_3/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_3/bincount/Const_2?
*category_encoding_3/bincount/DenseBincountDenseBincountAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_3/bincount/Minimum:z:0-category_encoding_3/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2,
*category_encoding_3/bincount/DenseBincount?
concatenate_7/PartitionedCallPartitionedCallnormalization_3/truediv:z:03category_encoding_3/bincount/DenseBincount:output:0*
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
I__inference_concatenate_7_layer_call_and_return_conditional_losses_2287482
concatenate_7/PartitionedCall?
IdentityIdentity&concatenate_7/PartitionedCall:output:0"^category_encoding_3/Assert/Assert'^normalization_3/Reshape/ReadVariableOp)^normalization_3/Reshape_1/ReadVariableOp9^string_lookup_3/None_lookup_table_find/LookupTableFindV2*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 2F
!category_encoding_3/Assert/Assert!category_encoding_3/Assert/Assert2P
&normalization_3/Reshape/ReadVariableOp&normalization_3/Reshape/ReadVariableOp2T
(normalization_3/Reshape_1/ReadVariableOp(normalization_3/Reshape_1/ReadVariableOp2t
8string_lookup_3/None_lookup_table_find/LookupTableFindV28string_lookup_3/None_lookup_table_find/LookupTableFindV2:O K
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
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:	

_output_shapes
: 
?
?
)__inference_dense_13_layer_call_fn_230280

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
GPU2*0J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_2290572
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
?
?
(__inference_model_7_layer_call_fn_229365
area
garden_area

month_sold
operating_cost	
rooms
square_meter
year_of_construction
	year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
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
month_soldoperating_costroomssquare_meteryear_of_construction	year_soldunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs


*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_7_layer_call_and_return_conditional_losses_2293382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 22
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
month_sold:WS
'
_output_shapes
:?????????
(
_user_specified_nameoperating_cost:NJ
'
_output_shapes
:?????????

_user_specified_namerooms:UQ
'
_output_shapes
:?????????
&
_user_specified_namesquare_meter:]Y
'
_output_shapes
:?????????
.
_user_specified_nameyear_of_construction:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:	

_output_shapes
: 
?
u
I__inference_concatenate_7_layer_call_and_return_conditional_losses_230230
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
':?????????:??????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?)
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_230112

inputs;
'dense_12_matmul_readvariableop_resource:
??7
(dense_12_biasadd_readvariableop_resource:	?:
'dense_13_matmul_readvariableop_resource:	?@6
(dense_13_biasadd_readvariableop_resource:@9
'dense_14_matmul_readvariableop_resource:@ 6
(dense_14_biasadd_readvariableop_resource: 9
'dense_15_matmul_readvariableop_resource: 6
(dense_15_biasadd_readvariableop_resource:
identity??dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?dense_14/BiasAdd/ReadVariableOp?dense_14/MatMul/ReadVariableOp?dense_15/BiasAdd/ReadVariableOp?dense_15/MatMul/ReadVariableOp?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMulinputs&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_12/BiasAdd}
dense_12/SigmoidSigmoiddense_12/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_12/Sigmoid?
dense_12/mulMuldense_12/BiasAdd:output:0dense_12/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
dense_12/mulw
dense_12/IdentityIdentitydense_12/mul:z:0*
T0*(
_output_shapes
:??????????2
dense_12/Identity?
dense_12/IdentityN	IdentityNdense_12/mul:z:0dense_12/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-230087*<
_output_shapes*
(:??????????:??????????2
dense_12/IdentityN?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02 
dense_13/MatMul/ReadVariableOp?
dense_13/MatMulMatMuldense_12/IdentityN:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_13/MatMul?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_13/BiasAdd?
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02 
dense_14/MatMul/ReadVariableOp?
dense_14/MatMulMatMuldense_13/BiasAdd:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_14/MatMul?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_14/BiasAdd/ReadVariableOp?
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_14/BiasAdd?
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_15/MatMul/ReadVariableOp?
dense_15/MatMulMatMuldense_14/BiasAdd:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_15/MatMul?
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp?
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_15/BiasAdd?
IdentityIdentitydense_15/BiasAdd:output:0 ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
-__inference_sequential_3_layer_call_fn_229115
dense_12_input
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
StatefulPartitionedCallStatefulPartitionedCalldense_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
GPU2*0J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_2290962
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
_user_specified_namedense_12_input
?
?
(__inference_model_6_layer_call_fn_230044
inputs_area
inputs_garden_area
inputs_month_sold
inputs_operating_cost
inputs_rooms
inputs_square_meter
inputs_year_of_construction
inputs_year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_areainputs_garden_areainputs_month_soldinputs_operating_costinputs_roomsinputs_square_meterinputs_year_of_constructioninputs_year_soldunknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
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
C__inference_model_6_layer_call_and_return_conditional_losses_2288692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 22
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
_user_specified_nameinputs/month_sold:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/operating_cost:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/rooms:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/square_meter:d`
'
_output_shapes
:?????????
5
_user_specified_nameinputs/year_of_construction:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/year_sold:	

_output_shapes
: 
?	
?
-__inference_sequential_3_layer_call_fn_230154

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
GPU2*0J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_2292022
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
?,
?
__inference_adapt_step_230223
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceIteratorGetNext:components:0moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
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

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype02
ReadVariableOpW
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:2
mula
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:2
mul_1P
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:2
add_1x
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1_
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:2
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
:2
powz
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype02
ReadVariableOp_2_
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:2
add_2N
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:2
mul_2_
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:2
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
:2
pow_1c
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:2
add_3R
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:2
mul_3R
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:2
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
?
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_229266
dense_12_input#
dense_12_229245:
??
dense_12_229247:	?"
dense_13_229250:	?@
dense_13_229252:@!
dense_14_229255:@ 
dense_14_229257: !
dense_15_229260: 
dense_15_229262:
identity?? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall? dense_14/StatefulPartitionedCall? dense_15/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCalldense_12_inputdense_12_229245dense_12_229247*
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
GPU2*0J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_2290412"
 dense_12/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_229250dense_13_229252*
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
GPU2*0J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_2290572"
 dense_13/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_229255dense_14_229257*
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
GPU2*0J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_2290732"
 dense_14/StatefulPartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_229260dense_15_229262*
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
GPU2*0J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_2290892"
 dense_15/StatefulPartitionedCall?
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall:X T
(
_output_shapes
:??????????
(
_user_specified_namedense_12_input
?
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_229290
dense_12_input#
dense_12_229269:
??
dense_12_229271:	?"
dense_13_229274:	?@
dense_13_229276:@!
dense_14_229279:@ 
dense_14_229281: !
dense_15_229284: 
dense_15_229286:
identity?? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall? dense_14/StatefulPartitionedCall? dense_15/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCalldense_12_inputdense_12_229269dense_12_229271*
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
GPU2*0J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_2290412"
 dense_12/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_229274dense_13_229276*
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
GPU2*0J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_2290572"
 dense_13/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_229279dense_14_229281*
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
GPU2*0J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_2290732"
 dense_14/StatefulPartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_229284dense_15_229286*
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
GPU2*0J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_2290892"
 dense_15/StatefulPartitionedCall?
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall:X T
(
_output_shapes
:??????????
(
_user_specified_namedense_12_input
?	
?
D__inference_dense_13_layer_call_and_return_conditional_losses_230271

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
?

?
I__inference_concatenate_6_layer_call_and_return_conditional_losses_230166
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:Q M
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
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6
?
?
)__inference_dense_14_layer_call_fn_230299

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
GPU2*0J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_2290732
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
?V
?
C__inference_model_6_layer_call_and_return_conditional_losses_230004
inputs_area
inputs_garden_area
inputs_month_sold
inputs_operating_cost
inputs_rooms
inputs_square_meter
inputs_year_of_construction
inputs_year_soldI
Estring_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value	=
/normalization_3_reshape_readvariableop_resource:?
1normalization_3_reshape_1_readvariableop_resource:
identity??!category_encoding_3/Assert/Assert?&normalization_3/Reshape/ReadVariableOp?(normalization_3/Reshape_1/ReadVariableOp?8string_lookup_3/None_lookup_table_find/LookupTableFindV2?
8string_lookup_3/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleinputs_areaFstring_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_3/None_lookup_table_find/LookupTableFindV2x
concatenate_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_6/concat/axis?
concatenate_6/concatConcatV2inputs_roomsinputs_square_meterinputs_garden_areainputs_operating_costinputs_year_of_constructioninputs_year_soldinputs_month_sold"concatenate_6/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate_6/concat?
&normalization_3/Reshape/ReadVariableOpReadVariableOp/normalization_3_reshape_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization_3/Reshape/ReadVariableOp?
normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_3/Reshape/shape?
normalization_3/ReshapeReshape.normalization_3/Reshape/ReadVariableOp:value:0&normalization_3/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_3/Reshape?
(normalization_3/Reshape_1/ReadVariableOpReadVariableOp1normalization_3_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02*
(normalization_3/Reshape_1/ReadVariableOp?
normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_3/Reshape_1/shape?
normalization_3/Reshape_1Reshape0normalization_3/Reshape_1/ReadVariableOp:value:0(normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_3/Reshape_1?
normalization_3/subSubconcatenate_6/concat:output:0 normalization_3/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization_3/sub?
normalization_3/SqrtSqrt"normalization_3/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_3/Maximum/y?
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_3/Maximum?
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_3/truediv?
category_encoding_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_3/Const?
category_encoding_3/MaxMaxAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0"category_encoding_3/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_3/Max?
category_encoding_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_3/Const_1?
category_encoding_3/MinMinAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0$category_encoding_3/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_3/Min{
category_encoding_3/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding_3/Cast/x?
category_encoding_3/CastCast#category_encoding_3/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_3/Cast?
category_encoding_3/GreaterGreatercategory_encoding_3/Cast:y:0 category_encoding_3/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_3/Greater~
category_encoding_3/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_3/Cast_1/x?
category_encoding_3/Cast_1Cast%category_encoding_3/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_3/Cast_1?
 category_encoding_3/GreaterEqualGreaterEqual category_encoding_3/Min:output:0category_encoding_3/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/GreaterEqual?
category_encoding_3/LogicalAnd
LogicalAndcategory_encoding_3/Greater:z:0$category_encoding_3/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_3/LogicalAnd?
 category_encoding_3/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=3002"
 category_encoding_3/Assert/Const?
(category_encoding_3/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=3002*
(category_encoding_3/Assert/Assert/data_0?
!category_encoding_3/Assert/AssertAssert"category_encoding_3/LogicalAnd:z:01category_encoding_3/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2#
!category_encoding_3/Assert/Assert?
"category_encoding_3/bincount/ShapeShapeAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_3/bincount/Shape?
"category_encoding_3/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_3/bincount/Const?
!category_encoding_3/bincount/ProdProd+category_encoding_3/bincount/Shape:output:0+category_encoding_3/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_3/bincount/Prod?
&category_encoding_3/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_3/bincount/Greater/y?
$category_encoding_3/bincount/GreaterGreater*category_encoding_3/bincount/Prod:output:0/category_encoding_3/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_3/bincount/Greater?
!category_encoding_3/bincount/CastCast(category_encoding_3/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_3/bincount/Cast?
$category_encoding_3/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_3/bincount/Const_1?
 category_encoding_3/bincount/MaxMaxAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_3/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/Max?
"category_encoding_3/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_3/bincount/add/y?
 category_encoding_3/bincount/addAddV2)category_encoding_3/bincount/Max:output:0+category_encoding_3/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/add?
 category_encoding_3/bincount/mulMul%category_encoding_3/bincount/Cast:y:0$category_encoding_3/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/mul?
&category_encoding_3/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_3/bincount/minlength?
$category_encoding_3/bincount/MaximumMaximum/category_encoding_3/bincount/minlength:output:0$category_encoding_3/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_3/bincount/Maximum?
&category_encoding_3/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_3/bincount/maxlength?
$category_encoding_3/bincount/MinimumMinimum/category_encoding_3/bincount/maxlength:output:0(category_encoding_3/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_3/bincount/Minimum?
$category_encoding_3/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_3/bincount/Const_2?
*category_encoding_3/bincount/DenseBincountDenseBincountAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_3/bincount/Minimum:z:0-category_encoding_3/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2,
*category_encoding_3/bincount/DenseBincountx
concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_7/concat/axis?
concatenate_7/concatConcatV2normalization_3/truediv:z:03category_encoding_3/bincount/DenseBincount:output:0"concatenate_7/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate_7/concat?
IdentityIdentityconcatenate_7/concat:output:0"^category_encoding_3/Assert/Assert'^normalization_3/Reshape/ReadVariableOp)^normalization_3/Reshape_1/ReadVariableOp9^string_lookup_3/None_lookup_table_find/LookupTableFindV2*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 2F
!category_encoding_3/Assert/Assert!category_encoding_3/Assert/Assert2P
&normalization_3/Reshape/ReadVariableOp&normalization_3/Reshape/ReadVariableOp2T
(normalization_3/Reshape_1/ReadVariableOp(normalization_3/Reshape_1/ReadVariableOp2t
8string_lookup_3/None_lookup_table_find/LookupTableFindV28string_lookup_3/None_lookup_table_find/LookupTableFindV2:T P
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
_user_specified_nameinputs/month_sold:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/operating_cost:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/rooms:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/square_meter:d`
'
_output_shapes
:?????????
5
_user_specified_nameinputs/year_of_construction:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/year_sold:	

_output_shapes
: 
?
+
__inference_<lambda>_230365
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
?
?
C__inference_model_7_layer_call_and_return_conditional_losses_229584
area
garden_area

month_sold
operating_cost	
rooms
square_meter
year_of_construction
	year_sold
model_6_229557
model_6_229559	
model_6_229561:
model_6_229563:'
sequential_3_229566:
??"
sequential_3_229568:	?&
sequential_3_229570:	?@!
sequential_3_229572:@%
sequential_3_229574:@ !
sequential_3_229576: %
sequential_3_229578: !
sequential_3_229580:
identity??model_6/StatefulPartitionedCall?$sequential_3/StatefulPartitionedCall?
model_6/StatefulPartitionedCallStatefulPartitionedCallareagarden_area
month_soldoperating_costroomssquare_meteryear_of_construction	year_soldmodel_6_229557model_6_229559model_6_229561model_6_229563*
Tin
2	*
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
C__inference_model_6_layer_call_and_return_conditional_losses_2288692!
model_6/StatefulPartitionedCall?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall(model_6/StatefulPartitionedCall:output:0sequential_3_229566sequential_3_229568sequential_3_229570sequential_3_229572sequential_3_229574sequential_3_229576sequential_3_229578sequential_3_229580*
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
GPU2*0J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_2292022&
$sequential_3/StatefulPartitionedCall?
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0 ^model_6/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2B
model_6/StatefulPartitionedCallmodel_6/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:M I
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
month_sold:WS
'
_output_shapes
:?????????
(
_user_specified_nameoperating_cost:NJ
'
_output_shapes
:?????????

_user_specified_namerooms:UQ
'
_output_shapes
:?????????
&
_user_specified_namesquare_meter:]Y
'
_output_shapes
:?????????
.
_user_specified_nameyear_of_construction:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:	

_output_shapes
: 
?	
?
D__inference_dense_15_layer_call_and_return_conditional_losses_229089

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
?
?
(__inference_model_6_layer_call_fn_230024
inputs_area
inputs_garden_area
inputs_month_sold
inputs_operating_cost
inputs_rooms
inputs_square_meter
inputs_year_of_construction
inputs_year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_areainputs_garden_areainputs_month_soldinputs_operating_costinputs_roomsinputs_square_meterinputs_year_of_constructioninputs_year_soldunknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
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
C__inference_model_6_layer_call_and_return_conditional_losses_2287512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 22
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
_user_specified_nameinputs/month_sold:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/operating_cost:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/rooms:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/square_meter:d`
'
_output_shapes
:?????????
5
_user_specified_nameinputs/year_of_construction:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/year_sold:	

_output_shapes
: 
?	
?
D__inference_dense_13_layer_call_and_return_conditional_losses_229057

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
?
?
(__inference_model_7_layer_call_fn_229510
area
garden_area

month_sold
operating_cost	
rooms
square_meter
year_of_construction
	year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
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
month_soldoperating_costroomssquare_meteryear_of_construction	year_soldunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs


*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_7_layer_call_and_return_conditional_losses_2294472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 22
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
month_sold:WS
'
_output_shapes
:?????????
(
_user_specified_nameoperating_cost:NJ
'
_output_shapes
:?????????

_user_specified_namerooms:UQ
'
_output_shapes
:?????????
&
_user_specified_namesquare_meter:]Y
'
_output_shapes
:?????????
.
_user_specified_nameyear_of_construction:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:	

_output_shapes
: 
?	
?
__inference_restore_fn_230360
restored_tensors_0
restored_tensors_1	N
Jstring_lookup_3_index_table_table_restore_lookuptableimportv2_table_handle
identity??=string_lookup_3_index_table_table_restore/LookupTableImportV2?
=string_lookup_3_index_table_table_restore/LookupTableImportV2LookupTableImportV2Jstring_lookup_3_index_table_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 2?
=string_lookup_3_index_table_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Const?
IdentityIdentityConst:output:0>^string_lookup_3_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2~
=string_lookup_3_index_table_table_restore/LookupTableImportV2=string_lookup_3_index_table_table_restore/LookupTableImportV2:L H
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
?
?
(__inference_model_6_layer_call_fn_228900
area
garden_area

month_sold
operating_cost	
rooms
square_meter
year_of_construction
	year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallareagarden_area
month_soldoperating_costroomssquare_meteryear_of_construction	year_soldunknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
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
C__inference_model_6_layer_call_and_return_conditional_losses_2288692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 22
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
month_sold:WS
'
_output_shapes
:?????????
(
_user_specified_nameoperating_cost:NJ
'
_output_shapes
:?????????

_user_specified_namerooms:UQ
'
_output_shapes
:?????????
&
_user_specified_namesquare_meter:]Y
'
_output_shapes
:?????????
.
_user_specified_nameyear_of_construction:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:	

_output_shapes
: 
?	
?
D__inference_dense_14_layer_call_and_return_conditional_losses_230290

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
?
?
C__inference_model_7_layer_call_and_return_conditional_losses_229547
area
garden_area

month_sold
operating_cost	
rooms
square_meter
year_of_construction
	year_sold
model_6_229520
model_6_229522	
model_6_229524:
model_6_229526:'
sequential_3_229529:
??"
sequential_3_229531:	?&
sequential_3_229533:	?@!
sequential_3_229535:@%
sequential_3_229537:@ !
sequential_3_229539: %
sequential_3_229541: !
sequential_3_229543:
identity??model_6/StatefulPartitionedCall?$sequential_3/StatefulPartitionedCall?
model_6/StatefulPartitionedCallStatefulPartitionedCallareagarden_area
month_soldoperating_costroomssquare_meteryear_of_construction	year_soldmodel_6_229520model_6_229522model_6_229524model_6_229526*
Tin
2	*
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
C__inference_model_6_layer_call_and_return_conditional_losses_2287512!
model_6/StatefulPartitionedCall?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall(model_6/StatefulPartitionedCall:output:0sequential_3_229529sequential_3_229531sequential_3_229533sequential_3_229535sequential_3_229537sequential_3_229539sequential_3_229541sequential_3_229543*
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
GPU2*0J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_2290962&
$sequential_3/StatefulPartitionedCall?
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0 ^model_6/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2B
model_6/StatefulPartitionedCallmodel_6/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:M I
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
month_sold:WS
'
_output_shapes
:?????????
(
_user_specified_nameoperating_cost:NJ
'
_output_shapes
:?????????

_user_specified_namerooms:UQ
'
_output_shapes
:?????????
&
_user_specified_namesquare_meter:]Y
'
_output_shapes
:?????????
.
_user_specified_nameyear_of_construction:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:	

_output_shapes
: 
?
Z
.__inference_concatenate_7_layer_call_fn_230236
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
I__inference_concatenate_7_layer_call_and_return_conditional_losses_2287482
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????:??????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?
?
__inference_save_fn_230352
checkpoint_key[
Wstring_lookup_3_index_table_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	??Jstring_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2?
Jstring_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Wstring_lookup_3_index_table_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::2L
Jstring_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2T
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
IdentityIdentityadd:z:0K^string_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2*
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

Identity_1IdentityConst:output:0K^string_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityQstring_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2:keys:0K^string_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
:2

Identity_2?

Identity_3Identity	add_1:z:0K^string_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2*
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

Identity_4IdentityConst_1:output:0K^string_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentitySstring_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2:values:0K^string_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2*
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
Jstring_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2Jstring_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?

?
I__inference_concatenate_6_layer_call_and_return_conditional_losses_228696

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:O K
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
 
_user_specified_nameinputs
?
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_229202

inputs#
dense_12_229181:
??
dense_12_229183:	?"
dense_13_229186:	?@
dense_13_229188:@!
dense_14_229191:@ 
dense_14_229193: !
dense_15_229196: 
dense_15_229198:
identity?? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall? dense_14/StatefulPartitionedCall? dense_15/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinputsdense_12_229181dense_12_229183*
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
GPU2*0J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_2290412"
 dense_12/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_229186dense_13_229188*
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
GPU2*0J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_2290572"
 dense_13/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_229191dense_14_229193*
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
GPU2*0J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_2290732"
 dense_14/StatefulPartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_229196dense_15_229198*
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
GPU2*0J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_2290892"
 dense_15/StatefulPartitionedCall?
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?)
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_230078

inputs;
'dense_12_matmul_readvariableop_resource:
??7
(dense_12_biasadd_readvariableop_resource:	?:
'dense_13_matmul_readvariableop_resource:	?@6
(dense_13_biasadd_readvariableop_resource:@9
'dense_14_matmul_readvariableop_resource:@ 6
(dense_14_biasadd_readvariableop_resource: 9
'dense_15_matmul_readvariableop_resource: 6
(dense_15_biasadd_readvariableop_resource:
identity??dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?dense_14/BiasAdd/ReadVariableOp?dense_14/MatMul/ReadVariableOp?dense_15/BiasAdd/ReadVariableOp?dense_15/MatMul/ReadVariableOp?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMulinputs&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_12/BiasAdd}
dense_12/SigmoidSigmoiddense_12/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_12/Sigmoid?
dense_12/mulMuldense_12/BiasAdd:output:0dense_12/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
dense_12/mulw
dense_12/IdentityIdentitydense_12/mul:z:0*
T0*(
_output_shapes
:??????????2
dense_12/Identity?
dense_12/IdentityN	IdentityNdense_12/mul:z:0dense_12/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-230053*<
_output_shapes*
(:??????????:??????????2
dense_12/IdentityN?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02 
dense_13/MatMul/ReadVariableOp?
dense_13/MatMulMatMuldense_12/IdentityN:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_13/MatMul?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_13/BiasAdd?
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02 
dense_14/MatMul/ReadVariableOp?
dense_14/MatMulMatMuldense_13/BiasAdd:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_14/MatMul?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_14/BiasAdd/ReadVariableOp?
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_14/BiasAdd?
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_15/MatMul/ReadVariableOp?
dense_15/MatMulMatMuldense_14/BiasAdd:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_15/MatMul?
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp?
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_15/BiasAdd?
IdentityIdentitydense_15/BiasAdd:output:0 ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_model_7_layer_call_and_return_conditional_losses_229447

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
model_6_229420
model_6_229422	
model_6_229424:
model_6_229426:'
sequential_3_229429:
??"
sequential_3_229431:	?&
sequential_3_229433:	?@!
sequential_3_229435:@%
sequential_3_229437:@ !
sequential_3_229439: %
sequential_3_229441: !
sequential_3_229443:
identity??model_6/StatefulPartitionedCall?$sequential_3/StatefulPartitionedCall?
model_6/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7model_6_229420model_6_229422model_6_229424model_6_229426*
Tin
2	*
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
C__inference_model_6_layer_call_and_return_conditional_losses_2288692!
model_6/StatefulPartitionedCall?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall(model_6/StatefulPartitionedCall:output:0sequential_3_229429sequential_3_229431sequential_3_229433sequential_3_229435sequential_3_229437sequential_3_229439sequential_3_229441sequential_3_229443*
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
GPU2*0J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_2292022&
$sequential_3/StatefulPartitionedCall?
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0 ^model_6/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2B
model_6/StatefulPartitionedCallmodel_6/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:O K
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
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:	

_output_shapes
: 
?
?
(__inference_model_7_layer_call_fn_229846
inputs_area
inputs_garden_area
inputs_month_sold
inputs_operating_cost
inputs_rooms
inputs_square_meter
inputs_year_of_construction
inputs_year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_areainputs_garden_areainputs_month_soldinputs_operating_costinputs_roomsinputs_square_meterinputs_year_of_constructioninputs_year_soldunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs


*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_7_layer_call_and_return_conditional_losses_2293382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 22
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
_user_specified_nameinputs/month_sold:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/operating_cost:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/rooms:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/square_meter:d`
'
_output_shapes
:?????????
5
_user_specified_nameinputs/year_of_construction:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/year_sold:	

_output_shapes
: 
?
?
.__inference_concatenate_6_layer_call_fn_230177
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_6_layer_call_and_return_conditional_losses_2286962
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:Q M
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
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6
?
?
(__inference_model_7_layer_call_fn_229882
inputs_area
inputs_garden_area
inputs_month_sold
inputs_operating_cost
inputs_rooms
inputs_square_meter
inputs_year_of_construction
inputs_year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_areainputs_garden_areainputs_month_soldinputs_operating_costinputs_roomsinputs_square_meterinputs_year_of_constructioninputs_year_soldunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs


*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_7_layer_call_and_return_conditional_losses_2294472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 22
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
_user_specified_nameinputs/month_sold:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/operating_cost:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/rooms:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/square_meter:d`
'
_output_shapes
:?????????
5
_user_specified_nameinputs/year_of_construction:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/year_sold:	

_output_shapes
: 
?	
?
-__inference_sequential_3_layer_call_fn_230133

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
GPU2*0J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_2290962
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
ؖ
?
"__inference__traced_restore_230619
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: %
assignvariableop_5_mean:)
assignvariableop_6_variance:"
assignvariableop_7_count:	 6
"assignvariableop_8_dense_12_kernel:
??/
 assignvariableop_9_dense_12_bias:	?6
#assignvariableop_10_dense_13_kernel:	?@/
!assignvariableop_11_dense_13_bias:@5
#assignvariableop_12_dense_14_kernel:@ /
!assignvariableop_13_dense_14_bias: 5
#assignvariableop_14_dense_15_kernel: /
!assignvariableop_15_dense_15_bias:c
Ystring_lookup_3_index_table_table_restore_lookuptableimportv2_string_lookup_3_index_table: #
assignvariableop_16_total: %
assignvariableop_17_count_1: >
*assignvariableop_18_adam_dense_12_kernel_m:
??7
(assignvariableop_19_adam_dense_12_bias_m:	?=
*assignvariableop_20_adam_dense_13_kernel_m:	?@6
(assignvariableop_21_adam_dense_13_bias_m:@<
*assignvariableop_22_adam_dense_14_kernel_m:@ 6
(assignvariableop_23_adam_dense_14_bias_m: <
*assignvariableop_24_adam_dense_15_kernel_m: 6
(assignvariableop_25_adam_dense_15_bias_m:>
*assignvariableop_26_adam_dense_12_kernel_v:
??7
(assignvariableop_27_adam_dense_12_bias_v:	?=
*assignvariableop_28_adam_dense_13_kernel_v:	?@6
(assignvariableop_29_adam_dense_13_bias_v:@<
*assignvariableop_30_adam_dense_14_kernel_v:@ 6
(assignvariableop_31_adam_dense_14_bias_v: <
*assignvariableop_32_adam_dense_15_kernel_v: 6
(assignvariableop_33_adam_dense_15_bias_v:
identity_35??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?=string_lookup_3_index_table_table_restore/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*?
value?B?%B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer_with_weights-0/_table/.ATTRIBUTES/table-keysBIlayer_with_weights-0/layer_with_weights-0/_table/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
'2%			2
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
AssignVariableOp_5AssignVariableOpassignvariableop_5_meanIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_varianceIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_countIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_12_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_12_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_13_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_13_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_14_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_14_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_15_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_15_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15?
=string_lookup_3_index_table_table_restore/LookupTableImportV2LookupTableImportV2Ystring_lookup_3_index_table_table_restore_lookuptableimportv2_string_lookup_3_index_tableRestoreV2:tensors:16RestoreV2:tensors:17*	
Tin0*

Tout0	*.
_class$
" loc:@string_lookup_3_index_table*
_output_shapes
 2?
=string_lookup_3_index_table_table_restore/LookupTableImportV2n
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
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_dense_12_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_dense_12_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_13_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_dense_13_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_14_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_dense_14_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_dense_15_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_dense_15_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_dense_12_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_dense_12_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_dense_13_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_dense_13_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_dense_14_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_dense_14_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_dense_15_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_dense_15_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_339
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp>^string_lookup_3_index_table_table_restore/LookupTableImportV2"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_34?
Identity_35IdentityIdentity_34:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9>^string_lookup_3_index_table_table_restore/LookupTableImportV2*
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
=string_lookup_3_index_table_table_restore/LookupTableImportV2=string_lookup_3_index_table_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:40
.
_class$
" loc:@string_lookup_3_index_table
?
?
)__inference_dense_15_layer_call_fn_230318

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
GPU2*0J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_2290892
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
??
?
C__inference_model_7_layer_call_and_return_conditional_losses_229719
inputs_area
inputs_garden_area
inputs_month_sold
inputs_operating_cost
inputs_rooms
inputs_square_meter
inputs_year_of_construction
inputs_year_soldQ
Mmodel_6_string_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleR
Nmodel_6_string_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value	E
7model_6_normalization_3_reshape_readvariableop_resource:G
9model_6_normalization_3_reshape_1_readvariableop_resource:H
4sequential_3_dense_12_matmul_readvariableop_resource:
??D
5sequential_3_dense_12_biasadd_readvariableop_resource:	?G
4sequential_3_dense_13_matmul_readvariableop_resource:	?@C
5sequential_3_dense_13_biasadd_readvariableop_resource:@F
4sequential_3_dense_14_matmul_readvariableop_resource:@ C
5sequential_3_dense_14_biasadd_readvariableop_resource: F
4sequential_3_dense_15_matmul_readvariableop_resource: C
5sequential_3_dense_15_biasadd_readvariableop_resource:
identity??)model_6/category_encoding_3/Assert/Assert?.model_6/normalization_3/Reshape/ReadVariableOp?0model_6/normalization_3/Reshape_1/ReadVariableOp?@model_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2?,sequential_3/dense_12/BiasAdd/ReadVariableOp?+sequential_3/dense_12/MatMul/ReadVariableOp?,sequential_3/dense_13/BiasAdd/ReadVariableOp?+sequential_3/dense_13/MatMul/ReadVariableOp?,sequential_3/dense_14/BiasAdd/ReadVariableOp?+sequential_3/dense_14/MatMul/ReadVariableOp?,sequential_3/dense_15/BiasAdd/ReadVariableOp?+sequential_3/dense_15/MatMul/ReadVariableOp?
@model_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Mmodel_6_string_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleinputs_areaNmodel_6_string_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2B
@model_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2?
!model_6/concatenate_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_6/concatenate_6/concat/axis?
model_6/concatenate_6/concatConcatV2inputs_roomsinputs_square_meterinputs_garden_areainputs_operating_costinputs_year_of_constructioninputs_year_soldinputs_month_sold*model_6/concatenate_6/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
model_6/concatenate_6/concat?
.model_6/normalization_3/Reshape/ReadVariableOpReadVariableOp7model_6_normalization_3_reshape_readvariableop_resource*
_output_shapes
:*
dtype020
.model_6/normalization_3/Reshape/ReadVariableOp?
%model_6/normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2'
%model_6/normalization_3/Reshape/shape?
model_6/normalization_3/ReshapeReshape6model_6/normalization_3/Reshape/ReadVariableOp:value:0.model_6/normalization_3/Reshape/shape:output:0*
T0*
_output_shapes

:2!
model_6/normalization_3/Reshape?
0model_6/normalization_3/Reshape_1/ReadVariableOpReadVariableOp9model_6_normalization_3_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype022
0model_6/normalization_3/Reshape_1/ReadVariableOp?
'model_6/normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2)
'model_6/normalization_3/Reshape_1/shape?
!model_6/normalization_3/Reshape_1Reshape8model_6/normalization_3/Reshape_1/ReadVariableOp:value:00model_6/normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes

:2#
!model_6/normalization_3/Reshape_1?
model_6/normalization_3/subSub%model_6/concatenate_6/concat:output:0(model_6/normalization_3/Reshape:output:0*
T0*'
_output_shapes
:?????????2
model_6/normalization_3/sub?
model_6/normalization_3/SqrtSqrt*model_6/normalization_3/Reshape_1:output:0*
T0*
_output_shapes

:2
model_6/normalization_3/Sqrt?
!model_6/normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32#
!model_6/normalization_3/Maximum/y?
model_6/normalization_3/MaximumMaximum model_6/normalization_3/Sqrt:y:0*model_6/normalization_3/Maximum/y:output:0*
T0*
_output_shapes

:2!
model_6/normalization_3/Maximum?
model_6/normalization_3/truedivRealDivmodel_6/normalization_3/sub:z:0#model_6/normalization_3/Maximum:z:0*
T0*'
_output_shapes
:?????????2!
model_6/normalization_3/truediv?
!model_6/category_encoding_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!model_6/category_encoding_3/Const?
model_6/category_encoding_3/MaxMaxImodel_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0*model_6/category_encoding_3/Const:output:0*
T0	*
_output_shapes
: 2!
model_6/category_encoding_3/Max?
#model_6/category_encoding_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#model_6/category_encoding_3/Const_1?
model_6/category_encoding_3/MinMinImodel_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0,model_6/category_encoding_3/Const_1:output:0*
T0	*
_output_shapes
: 2!
model_6/category_encoding_3/Min?
"model_6/category_encoding_3/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2$
"model_6/category_encoding_3/Cast/x?
 model_6/category_encoding_3/CastCast+model_6/category_encoding_3/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2"
 model_6/category_encoding_3/Cast?
#model_6/category_encoding_3/GreaterGreater$model_6/category_encoding_3/Cast:y:0(model_6/category_encoding_3/Max:output:0*
T0	*
_output_shapes
: 2%
#model_6/category_encoding_3/Greater?
$model_6/category_encoding_3/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2&
$model_6/category_encoding_3/Cast_1/x?
"model_6/category_encoding_3/Cast_1Cast-model_6/category_encoding_3/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2$
"model_6/category_encoding_3/Cast_1?
(model_6/category_encoding_3/GreaterEqualGreaterEqual(model_6/category_encoding_3/Min:output:0&model_6/category_encoding_3/Cast_1:y:0*
T0	*
_output_shapes
: 2*
(model_6/category_encoding_3/GreaterEqual?
&model_6/category_encoding_3/LogicalAnd
LogicalAnd'model_6/category_encoding_3/Greater:z:0,model_6/category_encoding_3/GreaterEqual:z:0*
_output_shapes
: 2(
&model_6/category_encoding_3/LogicalAnd?
(model_6/category_encoding_3/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=3002*
(model_6/category_encoding_3/Assert/Const?
0model_6/category_encoding_3/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=30022
0model_6/category_encoding_3/Assert/Assert/data_0?
)model_6/category_encoding_3/Assert/AssertAssert*model_6/category_encoding_3/LogicalAnd:z:09model_6/category_encoding_3/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2+
)model_6/category_encoding_3/Assert/Assert?
*model_6/category_encoding_3/bincount/ShapeShapeImodel_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2,
*model_6/category_encoding_3/bincount/Shape?
*model_6/category_encoding_3/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*model_6/category_encoding_3/bincount/Const?
)model_6/category_encoding_3/bincount/ProdProd3model_6/category_encoding_3/bincount/Shape:output:03model_6/category_encoding_3/bincount/Const:output:0*
T0*
_output_shapes
: 2+
)model_6/category_encoding_3/bincount/Prod?
.model_6/category_encoding_3/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 20
.model_6/category_encoding_3/bincount/Greater/y?
,model_6/category_encoding_3/bincount/GreaterGreater2model_6/category_encoding_3/bincount/Prod:output:07model_6/category_encoding_3/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2.
,model_6/category_encoding_3/bincount/Greater?
)model_6/category_encoding_3/bincount/CastCast0model_6/category_encoding_3/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2+
)model_6/category_encoding_3/bincount/Cast?
,model_6/category_encoding_3/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2.
,model_6/category_encoding_3/bincount/Const_1?
(model_6/category_encoding_3/bincount/MaxMaxImodel_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2:values:05model_6/category_encoding_3/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2*
(model_6/category_encoding_3/bincount/Max?
*model_6/category_encoding_3/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2,
*model_6/category_encoding_3/bincount/add/y?
(model_6/category_encoding_3/bincount/addAddV21model_6/category_encoding_3/bincount/Max:output:03model_6/category_encoding_3/bincount/add/y:output:0*
T0	*
_output_shapes
: 2*
(model_6/category_encoding_3/bincount/add?
(model_6/category_encoding_3/bincount/mulMul-model_6/category_encoding_3/bincount/Cast:y:0,model_6/category_encoding_3/bincount/add:z:0*
T0	*
_output_shapes
: 2*
(model_6/category_encoding_3/bincount/mul?
.model_6/category_encoding_3/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?20
.model_6/category_encoding_3/bincount/minlength?
,model_6/category_encoding_3/bincount/MaximumMaximum7model_6/category_encoding_3/bincount/minlength:output:0,model_6/category_encoding_3/bincount/mul:z:0*
T0	*
_output_shapes
: 2.
,model_6/category_encoding_3/bincount/Maximum?
.model_6/category_encoding_3/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?20
.model_6/category_encoding_3/bincount/maxlength?
,model_6/category_encoding_3/bincount/MinimumMinimum7model_6/category_encoding_3/bincount/maxlength:output:00model_6/category_encoding_3/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2.
,model_6/category_encoding_3/bincount/Minimum?
,model_6/category_encoding_3/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2.
,model_6/category_encoding_3/bincount/Const_2?
2model_6/category_encoding_3/bincount/DenseBincountDenseBincountImodel_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2:values:00model_6/category_encoding_3/bincount/Minimum:z:05model_6/category_encoding_3/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(24
2model_6/category_encoding_3/bincount/DenseBincount?
!model_6/concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_6/concatenate_7/concat/axis?
model_6/concatenate_7/concatConcatV2#model_6/normalization_3/truediv:z:0;model_6/category_encoding_3/bincount/DenseBincount:output:0*model_6/concatenate_7/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
model_6/concatenate_7/concat?
+sequential_3/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+sequential_3/dense_12/MatMul/ReadVariableOp?
sequential_3/dense_12/MatMulMatMul%model_6/concatenate_7/concat:output:03sequential_3/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_12/MatMul?
,sequential_3/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential_3/dense_12/BiasAdd/ReadVariableOp?
sequential_3/dense_12/BiasAddBiasAdd&sequential_3/dense_12/MatMul:product:04sequential_3/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_12/BiasAdd?
sequential_3/dense_12/SigmoidSigmoid&sequential_3/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_12/Sigmoid?
sequential_3/dense_12/mulMul&sequential_3/dense_12/BiasAdd:output:0!sequential_3/dense_12/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_12/mul?
sequential_3/dense_12/IdentityIdentitysequential_3/dense_12/mul:z:0*
T0*(
_output_shapes
:??????????2 
sequential_3/dense_12/Identity?
sequential_3/dense_12/IdentityN	IdentityNsequential_3/dense_12/mul:z:0&sequential_3/dense_12/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-229694*<
_output_shapes*
(:??????????:??????????2!
sequential_3/dense_12/IdentityN?
+sequential_3/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_13_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02-
+sequential_3/dense_13/MatMul/ReadVariableOp?
sequential_3/dense_13/MatMulMatMul(sequential_3/dense_12/IdentityN:output:03sequential_3/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential_3/dense_13/MatMul?
,sequential_3/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_3/dense_13/BiasAdd/ReadVariableOp?
sequential_3/dense_13/BiasAddBiasAdd&sequential_3/dense_13/MatMul:product:04sequential_3/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential_3/dense_13/BiasAdd?
+sequential_3/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_14_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02-
+sequential_3/dense_14/MatMul/ReadVariableOp?
sequential_3/dense_14/MatMulMatMul&sequential_3/dense_13/BiasAdd:output:03sequential_3/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_3/dense_14/MatMul?
,sequential_3/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/dense_14/BiasAdd/ReadVariableOp?
sequential_3/dense_14/BiasAddBiasAdd&sequential_3/dense_14/MatMul:product:04sequential_3/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_3/dense_14/BiasAdd?
+sequential_3/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_15_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+sequential_3/dense_15/MatMul/ReadVariableOp?
sequential_3/dense_15/MatMulMatMul&sequential_3/dense_14/BiasAdd:output:03sequential_3/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_3/dense_15/MatMul?
,sequential_3/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_3/dense_15/BiasAdd/ReadVariableOp?
sequential_3/dense_15/BiasAddBiasAdd&sequential_3/dense_15/MatMul:product:04sequential_3/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_3/dense_15/BiasAdd?
IdentityIdentity&sequential_3/dense_15/BiasAdd:output:0*^model_6/category_encoding_3/Assert/Assert/^model_6/normalization_3/Reshape/ReadVariableOp1^model_6/normalization_3/Reshape_1/ReadVariableOpA^model_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2-^sequential_3/dense_12/BiasAdd/ReadVariableOp,^sequential_3/dense_12/MatMul/ReadVariableOp-^sequential_3/dense_13/BiasAdd/ReadVariableOp,^sequential_3/dense_13/MatMul/ReadVariableOp-^sequential_3/dense_14/BiasAdd/ReadVariableOp,^sequential_3/dense_14/MatMul/ReadVariableOp-^sequential_3/dense_15/BiasAdd/ReadVariableOp,^sequential_3/dense_15/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2V
)model_6/category_encoding_3/Assert/Assert)model_6/category_encoding_3/Assert/Assert2`
.model_6/normalization_3/Reshape/ReadVariableOp.model_6/normalization_3/Reshape/ReadVariableOp2d
0model_6/normalization_3/Reshape_1/ReadVariableOp0model_6/normalization_3/Reshape_1/ReadVariableOp2?
@model_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2@model_6/string_lookup_3/None_lookup_table_find/LookupTableFindV22\
,sequential_3/dense_12/BiasAdd/ReadVariableOp,sequential_3/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_12/MatMul/ReadVariableOp+sequential_3/dense_12/MatMul/ReadVariableOp2\
,sequential_3/dense_13/BiasAdd/ReadVariableOp,sequential_3/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_13/MatMul/ReadVariableOp+sequential_3/dense_13/MatMul/ReadVariableOp2\
,sequential_3/dense_14/BiasAdd/ReadVariableOp,sequential_3/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_14/MatMul/ReadVariableOp+sequential_3/dense_14/MatMul/ReadVariableOp2\
,sequential_3/dense_15/BiasAdd/ReadVariableOp,sequential_3/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_15/MatMul/ReadVariableOp+sequential_3/dense_15/MatMul/ReadVariableOp:T P
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
_user_specified_nameinputs/month_sold:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/operating_cost:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/rooms:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/square_meter:d`
'
_output_shapes
:?????????
5
_user_specified_nameinputs/year_of_construction:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/year_sold:	

_output_shapes
: 
?
?
$__inference_signature_wrapper_229628
area
garden_area

month_sold
operating_cost	
rooms
square_meter
year_of_construction
	year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
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
month_soldoperating_costroomssquare_meteryear_of_construction	year_soldunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs


*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_2286602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 22
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
month_sold:WS
'
_output_shapes
:?????????
(
_user_specified_nameoperating_cost:NJ
'
_output_shapes
:?????????

_user_specified_namerooms:UQ
'
_output_shapes
:?????????
&
_user_specified_namesquare_meter:]Y
'
_output_shapes
:?????????
.
_user_specified_nameyear_of_construction:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:	

_output_shapes
: 
?	
?
-__inference_sequential_3_layer_call_fn_229242
dense_12_input
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
StatefulPartitionedCallStatefulPartitionedCalldense_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
GPU2*0J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_2292022
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
_user_specified_namedense_12_input
?	
?
D__inference_dense_15_layer_call_and_return_conditional_losses_230309

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
?
?
(__inference_model_6_layer_call_fn_228762
area
garden_area

month_sold
operating_cost	
rooms
square_meter
year_of_construction
	year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallareagarden_area
month_soldoperating_costroomssquare_meteryear_of_construction	year_soldunknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
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
C__inference_model_6_layer_call_and_return_conditional_losses_2287512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 22
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
month_sold:WS
'
_output_shapes
:?????????
(
_user_specified_nameoperating_cost:NJ
'
_output_shapes
:?????????

_user_specified_namerooms:UQ
'
_output_shapes
:?????????
&
_user_specified_namesquare_meter:]Y
'
_output_shapes
:?????????
.
_user_specified_nameyear_of_construction:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:	

_output_shapes
: 
?J
?
__inference__traced_save_230504
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableopU
Qsavev2_string_lookup_3_index_table_lookup_table_export_values_lookuptableexportv2W
Ssavev2_string_lookup_3_index_table_lookup_table_export_values_lookuptableexportv2_1	$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableop5
1savev2_adam_dense_13_kernel_m_read_readvariableop3
/savev2_adam_dense_13_bias_m_read_readvariableop5
1savev2_adam_dense_14_kernel_m_read_readvariableop3
/savev2_adam_dense_14_bias_m_read_readvariableop5
1savev2_adam_dense_15_kernel_m_read_readvariableop3
/savev2_adam_dense_15_bias_m_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableop5
1savev2_adam_dense_13_kernel_v_read_readvariableop3
/savev2_adam_dense_13_bias_v_read_readvariableop5
1savev2_adam_dense_14_kernel_v_read_readvariableop3
/savev2_adam_dense_14_bias_v_read_readvariableop5
1savev2_adam_dense_15_kernel_v_read_readvariableop3
/savev2_adam_dense_15_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*?
value?B?%B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer_with_weights-0/_table/.ATTRIBUTES/table-keysBIlayer_with_weights-0/layer_with_weights-0/_table/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopsavev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableopQsavev2_string_lookup_3_index_table_lookup_table_export_values_lookuptableexportv2Ssavev2_string_lookup_3_index_table_lookup_table_export_values_lookuptableexportv2_1 savev2_total_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop1savev2_adam_dense_13_kernel_m_read_readvariableop/savev2_adam_dense_13_bias_m_read_readvariableop1savev2_adam_dense_14_kernel_m_read_readvariableop/savev2_adam_dense_14_bias_m_read_readvariableop1savev2_adam_dense_15_kernel_m_read_readvariableop/savev2_adam_dense_15_bias_m_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableop1savev2_adam_dense_13_kernel_v_read_readvariableop/savev2_adam_dense_13_bias_v_read_readvariableop1savev2_adam_dense_14_kernel_v_read_readvariableop/savev2_adam_dense_14_bias_v_read_readvariableop1savev2_adam_dense_15_kernel_v_read_readvariableop/savev2_adam_dense_15_bias_v_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 *3
dtypes)
'2%			2
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
?: : : : : : ::: :
??:?:	?@:@:@ : : :::: : :
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
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::
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
?
/
__inference__initializer_230328
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
?U
?
C__inference_model_6_layer_call_and_return_conditional_losses_228869

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7I
Estring_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value	=
/normalization_3_reshape_readvariableop_resource:?
1normalization_3_reshape_1_readvariableop_resource:
identity??!category_encoding_3/Assert/Assert?&normalization_3/Reshape/ReadVariableOp?(normalization_3/Reshape_1/ReadVariableOp?8string_lookup_3/None_lookup_table_find/LookupTableFindV2?
8string_lookup_3/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleinputsFstring_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_3/None_lookup_table_find/LookupTableFindV2?
concatenate_6/PartitionedCallPartitionedCallinputs_4inputs_5inputs_1inputs_3inputs_6inputs_7inputs_2*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_6_layer_call_and_return_conditional_losses_2286962
concatenate_6/PartitionedCall?
&normalization_3/Reshape/ReadVariableOpReadVariableOp/normalization_3_reshape_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization_3/Reshape/ReadVariableOp?
normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_3/Reshape/shape?
normalization_3/ReshapeReshape.normalization_3/Reshape/ReadVariableOp:value:0&normalization_3/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_3/Reshape?
(normalization_3/Reshape_1/ReadVariableOpReadVariableOp1normalization_3_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02*
(normalization_3/Reshape_1/ReadVariableOp?
normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_3/Reshape_1/shape?
normalization_3/Reshape_1Reshape0normalization_3/Reshape_1/ReadVariableOp:value:0(normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_3/Reshape_1?
normalization_3/subSub&concatenate_6/PartitionedCall:output:0 normalization_3/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization_3/sub?
normalization_3/SqrtSqrt"normalization_3/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_3/Maximum/y?
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_3/Maximum?
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_3/truediv?
category_encoding_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_3/Const?
category_encoding_3/MaxMaxAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0"category_encoding_3/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_3/Max?
category_encoding_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_3/Const_1?
category_encoding_3/MinMinAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0$category_encoding_3/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_3/Min{
category_encoding_3/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding_3/Cast/x?
category_encoding_3/CastCast#category_encoding_3/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_3/Cast?
category_encoding_3/GreaterGreatercategory_encoding_3/Cast:y:0 category_encoding_3/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_3/Greater~
category_encoding_3/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_3/Cast_1/x?
category_encoding_3/Cast_1Cast%category_encoding_3/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_3/Cast_1?
 category_encoding_3/GreaterEqualGreaterEqual category_encoding_3/Min:output:0category_encoding_3/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/GreaterEqual?
category_encoding_3/LogicalAnd
LogicalAndcategory_encoding_3/Greater:z:0$category_encoding_3/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_3/LogicalAnd?
 category_encoding_3/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=3002"
 category_encoding_3/Assert/Const?
(category_encoding_3/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=3002*
(category_encoding_3/Assert/Assert/data_0?
!category_encoding_3/Assert/AssertAssert"category_encoding_3/LogicalAnd:z:01category_encoding_3/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2#
!category_encoding_3/Assert/Assert?
"category_encoding_3/bincount/ShapeShapeAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_3/bincount/Shape?
"category_encoding_3/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_3/bincount/Const?
!category_encoding_3/bincount/ProdProd+category_encoding_3/bincount/Shape:output:0+category_encoding_3/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_3/bincount/Prod?
&category_encoding_3/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_3/bincount/Greater/y?
$category_encoding_3/bincount/GreaterGreater*category_encoding_3/bincount/Prod:output:0/category_encoding_3/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_3/bincount/Greater?
!category_encoding_3/bincount/CastCast(category_encoding_3/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_3/bincount/Cast?
$category_encoding_3/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_3/bincount/Const_1?
 category_encoding_3/bincount/MaxMaxAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_3/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/Max?
"category_encoding_3/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_3/bincount/add/y?
 category_encoding_3/bincount/addAddV2)category_encoding_3/bincount/Max:output:0+category_encoding_3/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/add?
 category_encoding_3/bincount/mulMul%category_encoding_3/bincount/Cast:y:0$category_encoding_3/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/mul?
&category_encoding_3/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_3/bincount/minlength?
$category_encoding_3/bincount/MaximumMaximum/category_encoding_3/bincount/minlength:output:0$category_encoding_3/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_3/bincount/Maximum?
&category_encoding_3/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_3/bincount/maxlength?
$category_encoding_3/bincount/MinimumMinimum/category_encoding_3/bincount/maxlength:output:0(category_encoding_3/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_3/bincount/Minimum?
$category_encoding_3/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_3/bincount/Const_2?
*category_encoding_3/bincount/DenseBincountDenseBincountAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_3/bincount/Minimum:z:0-category_encoding_3/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2,
*category_encoding_3/bincount/DenseBincount?
concatenate_7/PartitionedCallPartitionedCallnormalization_3/truediv:z:03category_encoding_3/bincount/DenseBincount:output:0*
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
I__inference_concatenate_7_layer_call_and_return_conditional_losses_2287482
concatenate_7/PartitionedCall?
IdentityIdentity&concatenate_7/PartitionedCall:output:0"^category_encoding_3/Assert/Assert'^normalization_3/Reshape/ReadVariableOp)^normalization_3/Reshape_1/ReadVariableOp9^string_lookup_3/None_lookup_table_find/LookupTableFindV2*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 2F
!category_encoding_3/Assert/Assert!category_encoding_3/Assert/Assert2P
&normalization_3/Reshape/ReadVariableOp&normalization_3/Reshape/ReadVariableOp2T
(normalization_3/Reshape_1/ReadVariableOp(normalization_3/Reshape_1/ReadVariableOp2t
8string_lookup_3/None_lookup_table_find/LookupTableFindV28string_lookup_3/None_lookup_table_find/LookupTableFindV2:O K
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
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:	

_output_shapes
: 
?
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_229096

inputs#
dense_12_229042:
??
dense_12_229044:	?"
dense_13_229058:	?@
dense_13_229060:@!
dense_14_229074:@ 
dense_14_229076: !
dense_15_229090: 
dense_15_229092:
identity?? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall? dense_14/StatefulPartitionedCall? dense_15/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinputsdense_12_229042dense_12_229044*
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
GPU2*0J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_2290412"
 dense_12/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_229058dense_13_229060*
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
GPU2*0J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_2290572"
 dense_13/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_229074dense_14_229076*
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
GPU2*0J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_2290732"
 dense_14/StatefulPartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_229090dense_15_229092*
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
GPU2*0J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_2290892"
 dense_15/StatefulPartitionedCall?
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_14_layer_call_and_return_conditional_losses_229073

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
??
?
!__inference__wrapped_model_228660
area
garden_area

month_sold
operating_cost	
rooms
square_meter
year_of_construction
	year_soldY
Umodel_7_model_6_string_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleZ
Vmodel_7_model_6_string_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value	M
?model_7_model_6_normalization_3_reshape_readvariableop_resource:O
Amodel_7_model_6_normalization_3_reshape_1_readvariableop_resource:P
<model_7_sequential_3_dense_12_matmul_readvariableop_resource:
??L
=model_7_sequential_3_dense_12_biasadd_readvariableop_resource:	?O
<model_7_sequential_3_dense_13_matmul_readvariableop_resource:	?@K
=model_7_sequential_3_dense_13_biasadd_readvariableop_resource:@N
<model_7_sequential_3_dense_14_matmul_readvariableop_resource:@ K
=model_7_sequential_3_dense_14_biasadd_readvariableop_resource: N
<model_7_sequential_3_dense_15_matmul_readvariableop_resource: K
=model_7_sequential_3_dense_15_biasadd_readvariableop_resource:
identity??1model_7/model_6/category_encoding_3/Assert/Assert?6model_7/model_6/normalization_3/Reshape/ReadVariableOp?8model_7/model_6/normalization_3/Reshape_1/ReadVariableOp?Hmodel_7/model_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2?4model_7/sequential_3/dense_12/BiasAdd/ReadVariableOp?3model_7/sequential_3/dense_12/MatMul/ReadVariableOp?4model_7/sequential_3/dense_13/BiasAdd/ReadVariableOp?3model_7/sequential_3/dense_13/MatMul/ReadVariableOp?4model_7/sequential_3/dense_14/BiasAdd/ReadVariableOp?3model_7/sequential_3/dense_14/MatMul/ReadVariableOp?4model_7/sequential_3/dense_15/BiasAdd/ReadVariableOp?3model_7/sequential_3/dense_15/MatMul/ReadVariableOp?
Hmodel_7/model_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Umodel_7_model_6_string_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleareaVmodel_7_model_6_string_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2J
Hmodel_7/model_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2?
)model_7/model_6/concatenate_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2+
)model_7/model_6/concatenate_6/concat/axis?
$model_7/model_6/concatenate_6/concatConcatV2roomssquare_metergarden_areaoperating_costyear_of_construction	year_sold
month_sold2model_7/model_6/concatenate_6/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2&
$model_7/model_6/concatenate_6/concat?
6model_7/model_6/normalization_3/Reshape/ReadVariableOpReadVariableOp?model_7_model_6_normalization_3_reshape_readvariableop_resource*
_output_shapes
:*
dtype028
6model_7/model_6/normalization_3/Reshape/ReadVariableOp?
-model_7/model_6/normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2/
-model_7/model_6/normalization_3/Reshape/shape?
'model_7/model_6/normalization_3/ReshapeReshape>model_7/model_6/normalization_3/Reshape/ReadVariableOp:value:06model_7/model_6/normalization_3/Reshape/shape:output:0*
T0*
_output_shapes

:2)
'model_7/model_6/normalization_3/Reshape?
8model_7/model_6/normalization_3/Reshape_1/ReadVariableOpReadVariableOpAmodel_7_model_6_normalization_3_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02:
8model_7/model_6/normalization_3/Reshape_1/ReadVariableOp?
/model_7/model_6/normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      21
/model_7/model_6/normalization_3/Reshape_1/shape?
)model_7/model_6/normalization_3/Reshape_1Reshape@model_7/model_6/normalization_3/Reshape_1/ReadVariableOp:value:08model_7/model_6/normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes

:2+
)model_7/model_6/normalization_3/Reshape_1?
#model_7/model_6/normalization_3/subSub-model_7/model_6/concatenate_6/concat:output:00model_7/model_6/normalization_3/Reshape:output:0*
T0*'
_output_shapes
:?????????2%
#model_7/model_6/normalization_3/sub?
$model_7/model_6/normalization_3/SqrtSqrt2model_7/model_6/normalization_3/Reshape_1:output:0*
T0*
_output_shapes

:2&
$model_7/model_6/normalization_3/Sqrt?
)model_7/model_6/normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32+
)model_7/model_6/normalization_3/Maximum/y?
'model_7/model_6/normalization_3/MaximumMaximum(model_7/model_6/normalization_3/Sqrt:y:02model_7/model_6/normalization_3/Maximum/y:output:0*
T0*
_output_shapes

:2)
'model_7/model_6/normalization_3/Maximum?
'model_7/model_6/normalization_3/truedivRealDiv'model_7/model_6/normalization_3/sub:z:0+model_7/model_6/normalization_3/Maximum:z:0*
T0*'
_output_shapes
:?????????2)
'model_7/model_6/normalization_3/truediv?
)model_7/model_6/category_encoding_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)model_7/model_6/category_encoding_3/Const?
'model_7/model_6/category_encoding_3/MaxMaxQmodel_7/model_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2:values:02model_7/model_6/category_encoding_3/Const:output:0*
T0	*
_output_shapes
: 2)
'model_7/model_6/category_encoding_3/Max?
+model_7/model_6/category_encoding_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2-
+model_7/model_6/category_encoding_3/Const_1?
'model_7/model_6/category_encoding_3/MinMinQmodel_7/model_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2:values:04model_7/model_6/category_encoding_3/Const_1:output:0*
T0	*
_output_shapes
: 2)
'model_7/model_6/category_encoding_3/Min?
*model_7/model_6/category_encoding_3/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2,
*model_7/model_6/category_encoding_3/Cast/x?
(model_7/model_6/category_encoding_3/CastCast3model_7/model_6/category_encoding_3/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2*
(model_7/model_6/category_encoding_3/Cast?
+model_7/model_6/category_encoding_3/GreaterGreater,model_7/model_6/category_encoding_3/Cast:y:00model_7/model_6/category_encoding_3/Max:output:0*
T0	*
_output_shapes
: 2-
+model_7/model_6/category_encoding_3/Greater?
,model_7/model_6/category_encoding_3/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2.
,model_7/model_6/category_encoding_3/Cast_1/x?
*model_7/model_6/category_encoding_3/Cast_1Cast5model_7/model_6/category_encoding_3/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2,
*model_7/model_6/category_encoding_3/Cast_1?
0model_7/model_6/category_encoding_3/GreaterEqualGreaterEqual0model_7/model_6/category_encoding_3/Min:output:0.model_7/model_6/category_encoding_3/Cast_1:y:0*
T0	*
_output_shapes
: 22
0model_7/model_6/category_encoding_3/GreaterEqual?
.model_7/model_6/category_encoding_3/LogicalAnd
LogicalAnd/model_7/model_6/category_encoding_3/Greater:z:04model_7/model_6/category_encoding_3/GreaterEqual:z:0*
_output_shapes
: 20
.model_7/model_6/category_encoding_3/LogicalAnd?
0model_7/model_6/category_encoding_3/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=30022
0model_7/model_6/category_encoding_3/Assert/Const?
8model_7/model_6/category_encoding_3/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=3002:
8model_7/model_6/category_encoding_3/Assert/Assert/data_0?
1model_7/model_6/category_encoding_3/Assert/AssertAssert2model_7/model_6/category_encoding_3/LogicalAnd:z:0Amodel_7/model_6/category_encoding_3/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 23
1model_7/model_6/category_encoding_3/Assert/Assert?
2model_7/model_6/category_encoding_3/bincount/ShapeShapeQmodel_7/model_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:24
2model_7/model_6/category_encoding_3/bincount/Shape?
2model_7/model_6/category_encoding_3/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 24
2model_7/model_6/category_encoding_3/bincount/Const?
1model_7/model_6/category_encoding_3/bincount/ProdProd;model_7/model_6/category_encoding_3/bincount/Shape:output:0;model_7/model_6/category_encoding_3/bincount/Const:output:0*
T0*
_output_shapes
: 23
1model_7/model_6/category_encoding_3/bincount/Prod?
6model_7/model_6/category_encoding_3/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 28
6model_7/model_6/category_encoding_3/bincount/Greater/y?
4model_7/model_6/category_encoding_3/bincount/GreaterGreater:model_7/model_6/category_encoding_3/bincount/Prod:output:0?model_7/model_6/category_encoding_3/bincount/Greater/y:output:0*
T0*
_output_shapes
: 26
4model_7/model_6/category_encoding_3/bincount/Greater?
1model_7/model_6/category_encoding_3/bincount/CastCast8model_7/model_6/category_encoding_3/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 23
1model_7/model_6/category_encoding_3/bincount/Cast?
4model_7/model_6/category_encoding_3/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       26
4model_7/model_6/category_encoding_3/bincount/Const_1?
0model_7/model_6/category_encoding_3/bincount/MaxMaxQmodel_7/model_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0=model_7/model_6/category_encoding_3/bincount/Const_1:output:0*
T0	*
_output_shapes
: 22
0model_7/model_6/category_encoding_3/bincount/Max?
2model_7/model_6/category_encoding_3/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R24
2model_7/model_6/category_encoding_3/bincount/add/y?
0model_7/model_6/category_encoding_3/bincount/addAddV29model_7/model_6/category_encoding_3/bincount/Max:output:0;model_7/model_6/category_encoding_3/bincount/add/y:output:0*
T0	*
_output_shapes
: 22
0model_7/model_6/category_encoding_3/bincount/add?
0model_7/model_6/category_encoding_3/bincount/mulMul5model_7/model_6/category_encoding_3/bincount/Cast:y:04model_7/model_6/category_encoding_3/bincount/add:z:0*
T0	*
_output_shapes
: 22
0model_7/model_6/category_encoding_3/bincount/mul?
6model_7/model_6/category_encoding_3/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?28
6model_7/model_6/category_encoding_3/bincount/minlength?
4model_7/model_6/category_encoding_3/bincount/MaximumMaximum?model_7/model_6/category_encoding_3/bincount/minlength:output:04model_7/model_6/category_encoding_3/bincount/mul:z:0*
T0	*
_output_shapes
: 26
4model_7/model_6/category_encoding_3/bincount/Maximum?
6model_7/model_6/category_encoding_3/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?28
6model_7/model_6/category_encoding_3/bincount/maxlength?
4model_7/model_6/category_encoding_3/bincount/MinimumMinimum?model_7/model_6/category_encoding_3/bincount/maxlength:output:08model_7/model_6/category_encoding_3/bincount/Maximum:z:0*
T0	*
_output_shapes
: 26
4model_7/model_6/category_encoding_3/bincount/Minimum?
4model_7/model_6/category_encoding_3/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 26
4model_7/model_6/category_encoding_3/bincount/Const_2?
:model_7/model_6/category_encoding_3/bincount/DenseBincountDenseBincountQmodel_7/model_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2:values:08model_7/model_6/category_encoding_3/bincount/Minimum:z:0=model_7/model_6/category_encoding_3/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2<
:model_7/model_6/category_encoding_3/bincount/DenseBincount?
)model_7/model_6/concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2+
)model_7/model_6/concatenate_7/concat/axis?
$model_7/model_6/concatenate_7/concatConcatV2+model_7/model_6/normalization_3/truediv:z:0Cmodel_7/model_6/category_encoding_3/bincount/DenseBincount:output:02model_7/model_6/concatenate_7/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2&
$model_7/model_6/concatenate_7/concat?
3model_7/sequential_3/dense_12/MatMul/ReadVariableOpReadVariableOp<model_7_sequential_3_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype025
3model_7/sequential_3/dense_12/MatMul/ReadVariableOp?
$model_7/sequential_3/dense_12/MatMulMatMul-model_7/model_6/concatenate_7/concat:output:0;model_7/sequential_3/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$model_7/sequential_3/dense_12/MatMul?
4model_7/sequential_3/dense_12/BiasAdd/ReadVariableOpReadVariableOp=model_7_sequential_3_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype026
4model_7/sequential_3/dense_12/BiasAdd/ReadVariableOp?
%model_7/sequential_3/dense_12/BiasAddBiasAdd.model_7/sequential_3/dense_12/MatMul:product:0<model_7/sequential_3/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%model_7/sequential_3/dense_12/BiasAdd?
%model_7/sequential_3/dense_12/SigmoidSigmoid.model_7/sequential_3/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2'
%model_7/sequential_3/dense_12/Sigmoid?
!model_7/sequential_3/dense_12/mulMul.model_7/sequential_3/dense_12/BiasAdd:output:0)model_7/sequential_3/dense_12/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2#
!model_7/sequential_3/dense_12/mul?
&model_7/sequential_3/dense_12/IdentityIdentity%model_7/sequential_3/dense_12/mul:z:0*
T0*(
_output_shapes
:??????????2(
&model_7/sequential_3/dense_12/Identity?
'model_7/sequential_3/dense_12/IdentityN	IdentityN%model_7/sequential_3/dense_12/mul:z:0.model_7/sequential_3/dense_12/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-228635*<
_output_shapes*
(:??????????:??????????2)
'model_7/sequential_3/dense_12/IdentityN?
3model_7/sequential_3/dense_13/MatMul/ReadVariableOpReadVariableOp<model_7_sequential_3_dense_13_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype025
3model_7/sequential_3/dense_13/MatMul/ReadVariableOp?
$model_7/sequential_3/dense_13/MatMulMatMul0model_7/sequential_3/dense_12/IdentityN:output:0;model_7/sequential_3/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2&
$model_7/sequential_3/dense_13/MatMul?
4model_7/sequential_3/dense_13/BiasAdd/ReadVariableOpReadVariableOp=model_7_sequential_3_dense_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype026
4model_7/sequential_3/dense_13/BiasAdd/ReadVariableOp?
%model_7/sequential_3/dense_13/BiasAddBiasAdd.model_7/sequential_3/dense_13/MatMul:product:0<model_7/sequential_3/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2'
%model_7/sequential_3/dense_13/BiasAdd?
3model_7/sequential_3/dense_14/MatMul/ReadVariableOpReadVariableOp<model_7_sequential_3_dense_14_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype025
3model_7/sequential_3/dense_14/MatMul/ReadVariableOp?
$model_7/sequential_3/dense_14/MatMulMatMul.model_7/sequential_3/dense_13/BiasAdd:output:0;model_7/sequential_3/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2&
$model_7/sequential_3/dense_14/MatMul?
4model_7/sequential_3/dense_14/BiasAdd/ReadVariableOpReadVariableOp=model_7_sequential_3_dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype026
4model_7/sequential_3/dense_14/BiasAdd/ReadVariableOp?
%model_7/sequential_3/dense_14/BiasAddBiasAdd.model_7/sequential_3/dense_14/MatMul:product:0<model_7/sequential_3/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2'
%model_7/sequential_3/dense_14/BiasAdd?
3model_7/sequential_3/dense_15/MatMul/ReadVariableOpReadVariableOp<model_7_sequential_3_dense_15_matmul_readvariableop_resource*
_output_shapes

: *
dtype025
3model_7/sequential_3/dense_15/MatMul/ReadVariableOp?
$model_7/sequential_3/dense_15/MatMulMatMul.model_7/sequential_3/dense_14/BiasAdd:output:0;model_7/sequential_3/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2&
$model_7/sequential_3/dense_15/MatMul?
4model_7/sequential_3/dense_15/BiasAdd/ReadVariableOpReadVariableOp=model_7_sequential_3_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4model_7/sequential_3/dense_15/BiasAdd/ReadVariableOp?
%model_7/sequential_3/dense_15/BiasAddBiasAdd.model_7/sequential_3/dense_15/MatMul:product:0<model_7/sequential_3/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2'
%model_7/sequential_3/dense_15/BiasAdd?
IdentityIdentity.model_7/sequential_3/dense_15/BiasAdd:output:02^model_7/model_6/category_encoding_3/Assert/Assert7^model_7/model_6/normalization_3/Reshape/ReadVariableOp9^model_7/model_6/normalization_3/Reshape_1/ReadVariableOpI^model_7/model_6/string_lookup_3/None_lookup_table_find/LookupTableFindV25^model_7/sequential_3/dense_12/BiasAdd/ReadVariableOp4^model_7/sequential_3/dense_12/MatMul/ReadVariableOp5^model_7/sequential_3/dense_13/BiasAdd/ReadVariableOp4^model_7/sequential_3/dense_13/MatMul/ReadVariableOp5^model_7/sequential_3/dense_14/BiasAdd/ReadVariableOp4^model_7/sequential_3/dense_14/MatMul/ReadVariableOp5^model_7/sequential_3/dense_15/BiasAdd/ReadVariableOp4^model_7/sequential_3/dense_15/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2f
1model_7/model_6/category_encoding_3/Assert/Assert1model_7/model_6/category_encoding_3/Assert/Assert2p
6model_7/model_6/normalization_3/Reshape/ReadVariableOp6model_7/model_6/normalization_3/Reshape/ReadVariableOp2t
8model_7/model_6/normalization_3/Reshape_1/ReadVariableOp8model_7/model_6/normalization_3/Reshape_1/ReadVariableOp2?
Hmodel_7/model_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2Hmodel_7/model_6/string_lookup_3/None_lookup_table_find/LookupTableFindV22l
4model_7/sequential_3/dense_12/BiasAdd/ReadVariableOp4model_7/sequential_3/dense_12/BiasAdd/ReadVariableOp2j
3model_7/sequential_3/dense_12/MatMul/ReadVariableOp3model_7/sequential_3/dense_12/MatMul/ReadVariableOp2l
4model_7/sequential_3/dense_13/BiasAdd/ReadVariableOp4model_7/sequential_3/dense_13/BiasAdd/ReadVariableOp2j
3model_7/sequential_3/dense_13/MatMul/ReadVariableOp3model_7/sequential_3/dense_13/MatMul/ReadVariableOp2l
4model_7/sequential_3/dense_14/BiasAdd/ReadVariableOp4model_7/sequential_3/dense_14/BiasAdd/ReadVariableOp2j
3model_7/sequential_3/dense_14/MatMul/ReadVariableOp3model_7/sequential_3/dense_14/MatMul/ReadVariableOp2l
4model_7/sequential_3/dense_15/BiasAdd/ReadVariableOp4model_7/sequential_3/dense_15/BiasAdd/ReadVariableOp2j
3model_7/sequential_3/dense_15/MatMul/ReadVariableOp3model_7/sequential_3/dense_15/MatMul/ReadVariableOp:M I
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
month_sold:WS
'
_output_shapes
:?????????
(
_user_specified_nameoperating_cost:NJ
'
_output_shapes
:?????????

_user_specified_namerooms:UQ
'
_output_shapes
:?????????
&
_user_specified_namesquare_meter:]Y
'
_output_shapes
:?????????
.
_user_specified_nameyear_of_construction:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:	

_output_shapes
: 
?
?
D__inference_dense_12_layer_call_and_return_conditional_losses_229041

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
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
2*,
_gradient_op_typeCustomGradient-229034*<
_output_shapes*
(:??????????:??????????2
	IdentityN?

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?V
?
C__inference_model_6_layer_call_and_return_conditional_losses_229018
area
garden_area

month_sold
operating_cost	
rooms
square_meter
year_of_construction
	year_soldI
Estring_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value	=
/normalization_3_reshape_readvariableop_resource:?
1normalization_3_reshape_1_readvariableop_resource:
identity??!category_encoding_3/Assert/Assert?&normalization_3/Reshape/ReadVariableOp?(normalization_3/Reshape_1/ReadVariableOp?8string_lookup_3/None_lookup_table_find/LookupTableFindV2?
8string_lookup_3/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleareaFstring_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_3/None_lookup_table_find/LookupTableFindV2?
concatenate_6/PartitionedCallPartitionedCallroomssquare_metergarden_areaoperating_costyear_of_construction	year_sold
month_sold*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_6_layer_call_and_return_conditional_losses_2286962
concatenate_6/PartitionedCall?
&normalization_3/Reshape/ReadVariableOpReadVariableOp/normalization_3_reshape_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization_3/Reshape/ReadVariableOp?
normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_3/Reshape/shape?
normalization_3/ReshapeReshape.normalization_3/Reshape/ReadVariableOp:value:0&normalization_3/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_3/Reshape?
(normalization_3/Reshape_1/ReadVariableOpReadVariableOp1normalization_3_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02*
(normalization_3/Reshape_1/ReadVariableOp?
normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_3/Reshape_1/shape?
normalization_3/Reshape_1Reshape0normalization_3/Reshape_1/ReadVariableOp:value:0(normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_3/Reshape_1?
normalization_3/subSub&concatenate_6/PartitionedCall:output:0 normalization_3/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization_3/sub?
normalization_3/SqrtSqrt"normalization_3/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_3/Maximum/y?
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_3/Maximum?
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_3/truediv?
category_encoding_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_3/Const?
category_encoding_3/MaxMaxAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0"category_encoding_3/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_3/Max?
category_encoding_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_3/Const_1?
category_encoding_3/MinMinAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0$category_encoding_3/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_3/Min{
category_encoding_3/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding_3/Cast/x?
category_encoding_3/CastCast#category_encoding_3/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_3/Cast?
category_encoding_3/GreaterGreatercategory_encoding_3/Cast:y:0 category_encoding_3/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_3/Greater~
category_encoding_3/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_3/Cast_1/x?
category_encoding_3/Cast_1Cast%category_encoding_3/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_3/Cast_1?
 category_encoding_3/GreaterEqualGreaterEqual category_encoding_3/Min:output:0category_encoding_3/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/GreaterEqual?
category_encoding_3/LogicalAnd
LogicalAndcategory_encoding_3/Greater:z:0$category_encoding_3/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_3/LogicalAnd?
 category_encoding_3/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=3002"
 category_encoding_3/Assert/Const?
(category_encoding_3/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=3002*
(category_encoding_3/Assert/Assert/data_0?
!category_encoding_3/Assert/AssertAssert"category_encoding_3/LogicalAnd:z:01category_encoding_3/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2#
!category_encoding_3/Assert/Assert?
"category_encoding_3/bincount/ShapeShapeAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_3/bincount/Shape?
"category_encoding_3/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_3/bincount/Const?
!category_encoding_3/bincount/ProdProd+category_encoding_3/bincount/Shape:output:0+category_encoding_3/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_3/bincount/Prod?
&category_encoding_3/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_3/bincount/Greater/y?
$category_encoding_3/bincount/GreaterGreater*category_encoding_3/bincount/Prod:output:0/category_encoding_3/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_3/bincount/Greater?
!category_encoding_3/bincount/CastCast(category_encoding_3/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_3/bincount/Cast?
$category_encoding_3/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_3/bincount/Const_1?
 category_encoding_3/bincount/MaxMaxAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_3/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/Max?
"category_encoding_3/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_3/bincount/add/y?
 category_encoding_3/bincount/addAddV2)category_encoding_3/bincount/Max:output:0+category_encoding_3/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/add?
 category_encoding_3/bincount/mulMul%category_encoding_3/bincount/Cast:y:0$category_encoding_3/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/mul?
&category_encoding_3/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_3/bincount/minlength?
$category_encoding_3/bincount/MaximumMaximum/category_encoding_3/bincount/minlength:output:0$category_encoding_3/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_3/bincount/Maximum?
&category_encoding_3/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_3/bincount/maxlength?
$category_encoding_3/bincount/MinimumMinimum/category_encoding_3/bincount/maxlength:output:0(category_encoding_3/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_3/bincount/Minimum?
$category_encoding_3/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_3/bincount/Const_2?
*category_encoding_3/bincount/DenseBincountDenseBincountAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_3/bincount/Minimum:z:0-category_encoding_3/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2,
*category_encoding_3/bincount/DenseBincount?
concatenate_7/PartitionedCallPartitionedCallnormalization_3/truediv:z:03category_encoding_3/bincount/DenseBincount:output:0*
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
I__inference_concatenate_7_layer_call_and_return_conditional_losses_2287482
concatenate_7/PartitionedCall?
IdentityIdentity&concatenate_7/PartitionedCall:output:0"^category_encoding_3/Assert/Assert'^normalization_3/Reshape/ReadVariableOp)^normalization_3/Reshape_1/ReadVariableOp9^string_lookup_3/None_lookup_table_find/LookupTableFindV2*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 2F
!category_encoding_3/Assert/Assert!category_encoding_3/Assert/Assert2P
&normalization_3/Reshape/ReadVariableOp&normalization_3/Reshape/ReadVariableOp2T
(normalization_3/Reshape_1/ReadVariableOp(normalization_3/Reshape_1/ReadVariableOp2t
8string_lookup_3/None_lookup_table_find/LookupTableFindV28string_lookup_3/None_lookup_table_find/LookupTableFindV2:M I
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
month_sold:WS
'
_output_shapes
:?????????
(
_user_specified_nameoperating_cost:NJ
'
_output_shapes
:?????????

_user_specified_namerooms:UQ
'
_output_shapes
:?????????
&
_user_specified_namesquare_meter:]Y
'
_output_shapes
:?????????
.
_user_specified_nameyear_of_construction:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:	

_output_shapes
: 
??
?
C__inference_model_7_layer_call_and_return_conditional_losses_229810
inputs_area
inputs_garden_area
inputs_month_sold
inputs_operating_cost
inputs_rooms
inputs_square_meter
inputs_year_of_construction
inputs_year_soldQ
Mmodel_6_string_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleR
Nmodel_6_string_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value	E
7model_6_normalization_3_reshape_readvariableop_resource:G
9model_6_normalization_3_reshape_1_readvariableop_resource:H
4sequential_3_dense_12_matmul_readvariableop_resource:
??D
5sequential_3_dense_12_biasadd_readvariableop_resource:	?G
4sequential_3_dense_13_matmul_readvariableop_resource:	?@C
5sequential_3_dense_13_biasadd_readvariableop_resource:@F
4sequential_3_dense_14_matmul_readvariableop_resource:@ C
5sequential_3_dense_14_biasadd_readvariableop_resource: F
4sequential_3_dense_15_matmul_readvariableop_resource: C
5sequential_3_dense_15_biasadd_readvariableop_resource:
identity??)model_6/category_encoding_3/Assert/Assert?.model_6/normalization_3/Reshape/ReadVariableOp?0model_6/normalization_3/Reshape_1/ReadVariableOp?@model_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2?,sequential_3/dense_12/BiasAdd/ReadVariableOp?+sequential_3/dense_12/MatMul/ReadVariableOp?,sequential_3/dense_13/BiasAdd/ReadVariableOp?+sequential_3/dense_13/MatMul/ReadVariableOp?,sequential_3/dense_14/BiasAdd/ReadVariableOp?+sequential_3/dense_14/MatMul/ReadVariableOp?,sequential_3/dense_15/BiasAdd/ReadVariableOp?+sequential_3/dense_15/MatMul/ReadVariableOp?
@model_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Mmodel_6_string_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleinputs_areaNmodel_6_string_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2B
@model_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2?
!model_6/concatenate_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_6/concatenate_6/concat/axis?
model_6/concatenate_6/concatConcatV2inputs_roomsinputs_square_meterinputs_garden_areainputs_operating_costinputs_year_of_constructioninputs_year_soldinputs_month_sold*model_6/concatenate_6/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
model_6/concatenate_6/concat?
.model_6/normalization_3/Reshape/ReadVariableOpReadVariableOp7model_6_normalization_3_reshape_readvariableop_resource*
_output_shapes
:*
dtype020
.model_6/normalization_3/Reshape/ReadVariableOp?
%model_6/normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2'
%model_6/normalization_3/Reshape/shape?
model_6/normalization_3/ReshapeReshape6model_6/normalization_3/Reshape/ReadVariableOp:value:0.model_6/normalization_3/Reshape/shape:output:0*
T0*
_output_shapes

:2!
model_6/normalization_3/Reshape?
0model_6/normalization_3/Reshape_1/ReadVariableOpReadVariableOp9model_6_normalization_3_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype022
0model_6/normalization_3/Reshape_1/ReadVariableOp?
'model_6/normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2)
'model_6/normalization_3/Reshape_1/shape?
!model_6/normalization_3/Reshape_1Reshape8model_6/normalization_3/Reshape_1/ReadVariableOp:value:00model_6/normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes

:2#
!model_6/normalization_3/Reshape_1?
model_6/normalization_3/subSub%model_6/concatenate_6/concat:output:0(model_6/normalization_3/Reshape:output:0*
T0*'
_output_shapes
:?????????2
model_6/normalization_3/sub?
model_6/normalization_3/SqrtSqrt*model_6/normalization_3/Reshape_1:output:0*
T0*
_output_shapes

:2
model_6/normalization_3/Sqrt?
!model_6/normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32#
!model_6/normalization_3/Maximum/y?
model_6/normalization_3/MaximumMaximum model_6/normalization_3/Sqrt:y:0*model_6/normalization_3/Maximum/y:output:0*
T0*
_output_shapes

:2!
model_6/normalization_3/Maximum?
model_6/normalization_3/truedivRealDivmodel_6/normalization_3/sub:z:0#model_6/normalization_3/Maximum:z:0*
T0*'
_output_shapes
:?????????2!
model_6/normalization_3/truediv?
!model_6/category_encoding_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!model_6/category_encoding_3/Const?
model_6/category_encoding_3/MaxMaxImodel_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0*model_6/category_encoding_3/Const:output:0*
T0	*
_output_shapes
: 2!
model_6/category_encoding_3/Max?
#model_6/category_encoding_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#model_6/category_encoding_3/Const_1?
model_6/category_encoding_3/MinMinImodel_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0,model_6/category_encoding_3/Const_1:output:0*
T0	*
_output_shapes
: 2!
model_6/category_encoding_3/Min?
"model_6/category_encoding_3/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2$
"model_6/category_encoding_3/Cast/x?
 model_6/category_encoding_3/CastCast+model_6/category_encoding_3/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2"
 model_6/category_encoding_3/Cast?
#model_6/category_encoding_3/GreaterGreater$model_6/category_encoding_3/Cast:y:0(model_6/category_encoding_3/Max:output:0*
T0	*
_output_shapes
: 2%
#model_6/category_encoding_3/Greater?
$model_6/category_encoding_3/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2&
$model_6/category_encoding_3/Cast_1/x?
"model_6/category_encoding_3/Cast_1Cast-model_6/category_encoding_3/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2$
"model_6/category_encoding_3/Cast_1?
(model_6/category_encoding_3/GreaterEqualGreaterEqual(model_6/category_encoding_3/Min:output:0&model_6/category_encoding_3/Cast_1:y:0*
T0	*
_output_shapes
: 2*
(model_6/category_encoding_3/GreaterEqual?
&model_6/category_encoding_3/LogicalAnd
LogicalAnd'model_6/category_encoding_3/Greater:z:0,model_6/category_encoding_3/GreaterEqual:z:0*
_output_shapes
: 2(
&model_6/category_encoding_3/LogicalAnd?
(model_6/category_encoding_3/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=3002*
(model_6/category_encoding_3/Assert/Const?
0model_6/category_encoding_3/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=30022
0model_6/category_encoding_3/Assert/Assert/data_0?
)model_6/category_encoding_3/Assert/AssertAssert*model_6/category_encoding_3/LogicalAnd:z:09model_6/category_encoding_3/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2+
)model_6/category_encoding_3/Assert/Assert?
*model_6/category_encoding_3/bincount/ShapeShapeImodel_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2,
*model_6/category_encoding_3/bincount/Shape?
*model_6/category_encoding_3/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*model_6/category_encoding_3/bincount/Const?
)model_6/category_encoding_3/bincount/ProdProd3model_6/category_encoding_3/bincount/Shape:output:03model_6/category_encoding_3/bincount/Const:output:0*
T0*
_output_shapes
: 2+
)model_6/category_encoding_3/bincount/Prod?
.model_6/category_encoding_3/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 20
.model_6/category_encoding_3/bincount/Greater/y?
,model_6/category_encoding_3/bincount/GreaterGreater2model_6/category_encoding_3/bincount/Prod:output:07model_6/category_encoding_3/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2.
,model_6/category_encoding_3/bincount/Greater?
)model_6/category_encoding_3/bincount/CastCast0model_6/category_encoding_3/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2+
)model_6/category_encoding_3/bincount/Cast?
,model_6/category_encoding_3/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2.
,model_6/category_encoding_3/bincount/Const_1?
(model_6/category_encoding_3/bincount/MaxMaxImodel_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2:values:05model_6/category_encoding_3/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2*
(model_6/category_encoding_3/bincount/Max?
*model_6/category_encoding_3/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2,
*model_6/category_encoding_3/bincount/add/y?
(model_6/category_encoding_3/bincount/addAddV21model_6/category_encoding_3/bincount/Max:output:03model_6/category_encoding_3/bincount/add/y:output:0*
T0	*
_output_shapes
: 2*
(model_6/category_encoding_3/bincount/add?
(model_6/category_encoding_3/bincount/mulMul-model_6/category_encoding_3/bincount/Cast:y:0,model_6/category_encoding_3/bincount/add:z:0*
T0	*
_output_shapes
: 2*
(model_6/category_encoding_3/bincount/mul?
.model_6/category_encoding_3/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?20
.model_6/category_encoding_3/bincount/minlength?
,model_6/category_encoding_3/bincount/MaximumMaximum7model_6/category_encoding_3/bincount/minlength:output:0,model_6/category_encoding_3/bincount/mul:z:0*
T0	*
_output_shapes
: 2.
,model_6/category_encoding_3/bincount/Maximum?
.model_6/category_encoding_3/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?20
.model_6/category_encoding_3/bincount/maxlength?
,model_6/category_encoding_3/bincount/MinimumMinimum7model_6/category_encoding_3/bincount/maxlength:output:00model_6/category_encoding_3/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2.
,model_6/category_encoding_3/bincount/Minimum?
,model_6/category_encoding_3/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2.
,model_6/category_encoding_3/bincount/Const_2?
2model_6/category_encoding_3/bincount/DenseBincountDenseBincountImodel_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2:values:00model_6/category_encoding_3/bincount/Minimum:z:05model_6/category_encoding_3/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(24
2model_6/category_encoding_3/bincount/DenseBincount?
!model_6/concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_6/concatenate_7/concat/axis?
model_6/concatenate_7/concatConcatV2#model_6/normalization_3/truediv:z:0;model_6/category_encoding_3/bincount/DenseBincount:output:0*model_6/concatenate_7/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
model_6/concatenate_7/concat?
+sequential_3/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+sequential_3/dense_12/MatMul/ReadVariableOp?
sequential_3/dense_12/MatMulMatMul%model_6/concatenate_7/concat:output:03sequential_3/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_12/MatMul?
,sequential_3/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential_3/dense_12/BiasAdd/ReadVariableOp?
sequential_3/dense_12/BiasAddBiasAdd&sequential_3/dense_12/MatMul:product:04sequential_3/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_12/BiasAdd?
sequential_3/dense_12/SigmoidSigmoid&sequential_3/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_12/Sigmoid?
sequential_3/dense_12/mulMul&sequential_3/dense_12/BiasAdd:output:0!sequential_3/dense_12/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_12/mul?
sequential_3/dense_12/IdentityIdentitysequential_3/dense_12/mul:z:0*
T0*(
_output_shapes
:??????????2 
sequential_3/dense_12/Identity?
sequential_3/dense_12/IdentityN	IdentityNsequential_3/dense_12/mul:z:0&sequential_3/dense_12/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-229785*<
_output_shapes*
(:??????????:??????????2!
sequential_3/dense_12/IdentityN?
+sequential_3/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_13_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02-
+sequential_3/dense_13/MatMul/ReadVariableOp?
sequential_3/dense_13/MatMulMatMul(sequential_3/dense_12/IdentityN:output:03sequential_3/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential_3/dense_13/MatMul?
,sequential_3/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_3/dense_13/BiasAdd/ReadVariableOp?
sequential_3/dense_13/BiasAddBiasAdd&sequential_3/dense_13/MatMul:product:04sequential_3/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential_3/dense_13/BiasAdd?
+sequential_3/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_14_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02-
+sequential_3/dense_14/MatMul/ReadVariableOp?
sequential_3/dense_14/MatMulMatMul&sequential_3/dense_13/BiasAdd:output:03sequential_3/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_3/dense_14/MatMul?
,sequential_3/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/dense_14/BiasAdd/ReadVariableOp?
sequential_3/dense_14/BiasAddBiasAdd&sequential_3/dense_14/MatMul:product:04sequential_3/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_3/dense_14/BiasAdd?
+sequential_3/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_15_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+sequential_3/dense_15/MatMul/ReadVariableOp?
sequential_3/dense_15/MatMulMatMul&sequential_3/dense_14/BiasAdd:output:03sequential_3/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_3/dense_15/MatMul?
,sequential_3/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_3/dense_15/BiasAdd/ReadVariableOp?
sequential_3/dense_15/BiasAddBiasAdd&sequential_3/dense_15/MatMul:product:04sequential_3/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_3/dense_15/BiasAdd?
IdentityIdentity&sequential_3/dense_15/BiasAdd:output:0*^model_6/category_encoding_3/Assert/Assert/^model_6/normalization_3/Reshape/ReadVariableOp1^model_6/normalization_3/Reshape_1/ReadVariableOpA^model_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2-^sequential_3/dense_12/BiasAdd/ReadVariableOp,^sequential_3/dense_12/MatMul/ReadVariableOp-^sequential_3/dense_13/BiasAdd/ReadVariableOp,^sequential_3/dense_13/MatMul/ReadVariableOp-^sequential_3/dense_14/BiasAdd/ReadVariableOp,^sequential_3/dense_14/MatMul/ReadVariableOp-^sequential_3/dense_15/BiasAdd/ReadVariableOp,^sequential_3/dense_15/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2V
)model_6/category_encoding_3/Assert/Assert)model_6/category_encoding_3/Assert/Assert2`
.model_6/normalization_3/Reshape/ReadVariableOp.model_6/normalization_3/Reshape/ReadVariableOp2d
0model_6/normalization_3/Reshape_1/ReadVariableOp0model_6/normalization_3/Reshape_1/ReadVariableOp2?
@model_6/string_lookup_3/None_lookup_table_find/LookupTableFindV2@model_6/string_lookup_3/None_lookup_table_find/LookupTableFindV22\
,sequential_3/dense_12/BiasAdd/ReadVariableOp,sequential_3/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_12/MatMul/ReadVariableOp+sequential_3/dense_12/MatMul/ReadVariableOp2\
,sequential_3/dense_13/BiasAdd/ReadVariableOp,sequential_3/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_13/MatMul/ReadVariableOp+sequential_3/dense_13/MatMul/ReadVariableOp2\
,sequential_3/dense_14/BiasAdd/ReadVariableOp,sequential_3/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_14/MatMul/ReadVariableOp+sequential_3/dense_14/MatMul/ReadVariableOp2\
,sequential_3/dense_15/BiasAdd/ReadVariableOp,sequential_3/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_15/MatMul/ReadVariableOp+sequential_3/dense_15/MatMul/ReadVariableOp:T P
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
_user_specified_nameinputs/month_sold:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/operating_cost:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/rooms:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/square_meter:d`
'
_output_shapes
:?????????
5
_user_specified_nameinputs/year_of_construction:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/year_sold:	

_output_shapes
: 
?
?
)__inference_dense_12_layer_call_fn_230261

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
GPU2*0J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_2290412
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
?V
?
C__inference_model_6_layer_call_and_return_conditional_losses_228959
area
garden_area

month_sold
operating_cost	
rooms
square_meter
year_of_construction
	year_soldI
Estring_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value	=
/normalization_3_reshape_readvariableop_resource:?
1normalization_3_reshape_1_readvariableop_resource:
identity??!category_encoding_3/Assert/Assert?&normalization_3/Reshape/ReadVariableOp?(normalization_3/Reshape_1/ReadVariableOp?8string_lookup_3/None_lookup_table_find/LookupTableFindV2?
8string_lookup_3/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleareaFstring_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_3/None_lookup_table_find/LookupTableFindV2?
concatenate_6/PartitionedCallPartitionedCallroomssquare_metergarden_areaoperating_costyear_of_construction	year_sold
month_sold*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_6_layer_call_and_return_conditional_losses_2286962
concatenate_6/PartitionedCall?
&normalization_3/Reshape/ReadVariableOpReadVariableOp/normalization_3_reshape_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization_3/Reshape/ReadVariableOp?
normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_3/Reshape/shape?
normalization_3/ReshapeReshape.normalization_3/Reshape/ReadVariableOp:value:0&normalization_3/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_3/Reshape?
(normalization_3/Reshape_1/ReadVariableOpReadVariableOp1normalization_3_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02*
(normalization_3/Reshape_1/ReadVariableOp?
normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_3/Reshape_1/shape?
normalization_3/Reshape_1Reshape0normalization_3/Reshape_1/ReadVariableOp:value:0(normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_3/Reshape_1?
normalization_3/subSub&concatenate_6/PartitionedCall:output:0 normalization_3/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization_3/sub?
normalization_3/SqrtSqrt"normalization_3/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_3/Maximum/y?
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_3/Maximum?
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_3/truediv?
category_encoding_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_3/Const?
category_encoding_3/MaxMaxAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0"category_encoding_3/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_3/Max?
category_encoding_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_3/Const_1?
category_encoding_3/MinMinAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0$category_encoding_3/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_3/Min{
category_encoding_3/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding_3/Cast/x?
category_encoding_3/CastCast#category_encoding_3/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_3/Cast?
category_encoding_3/GreaterGreatercategory_encoding_3/Cast:y:0 category_encoding_3/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_3/Greater~
category_encoding_3/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_3/Cast_1/x?
category_encoding_3/Cast_1Cast%category_encoding_3/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_3/Cast_1?
 category_encoding_3/GreaterEqualGreaterEqual category_encoding_3/Min:output:0category_encoding_3/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/GreaterEqual?
category_encoding_3/LogicalAnd
LogicalAndcategory_encoding_3/Greater:z:0$category_encoding_3/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_3/LogicalAnd?
 category_encoding_3/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=3002"
 category_encoding_3/Assert/Const?
(category_encoding_3/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=3002*
(category_encoding_3/Assert/Assert/data_0?
!category_encoding_3/Assert/AssertAssert"category_encoding_3/LogicalAnd:z:01category_encoding_3/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2#
!category_encoding_3/Assert/Assert?
"category_encoding_3/bincount/ShapeShapeAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_3/bincount/Shape?
"category_encoding_3/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_3/bincount/Const?
!category_encoding_3/bincount/ProdProd+category_encoding_3/bincount/Shape:output:0+category_encoding_3/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_3/bincount/Prod?
&category_encoding_3/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_3/bincount/Greater/y?
$category_encoding_3/bincount/GreaterGreater*category_encoding_3/bincount/Prod:output:0/category_encoding_3/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_3/bincount/Greater?
!category_encoding_3/bincount/CastCast(category_encoding_3/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_3/bincount/Cast?
$category_encoding_3/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_3/bincount/Const_1?
 category_encoding_3/bincount/MaxMaxAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_3/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/Max?
"category_encoding_3/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_3/bincount/add/y?
 category_encoding_3/bincount/addAddV2)category_encoding_3/bincount/Max:output:0+category_encoding_3/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/add?
 category_encoding_3/bincount/mulMul%category_encoding_3/bincount/Cast:y:0$category_encoding_3/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/mul?
&category_encoding_3/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_3/bincount/minlength?
$category_encoding_3/bincount/MaximumMaximum/category_encoding_3/bincount/minlength:output:0$category_encoding_3/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_3/bincount/Maximum?
&category_encoding_3/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_3/bincount/maxlength?
$category_encoding_3/bincount/MinimumMinimum/category_encoding_3/bincount/maxlength:output:0(category_encoding_3/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_3/bincount/Minimum?
$category_encoding_3/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_3/bincount/Const_2?
*category_encoding_3/bincount/DenseBincountDenseBincountAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_3/bincount/Minimum:z:0-category_encoding_3/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2,
*category_encoding_3/bincount/DenseBincount?
concatenate_7/PartitionedCallPartitionedCallnormalization_3/truediv:z:03category_encoding_3/bincount/DenseBincount:output:0*
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
I__inference_concatenate_7_layer_call_and_return_conditional_losses_2287482
concatenate_7/PartitionedCall?
IdentityIdentity&concatenate_7/PartitionedCall:output:0"^category_encoding_3/Assert/Assert'^normalization_3/Reshape/ReadVariableOp)^normalization_3/Reshape_1/ReadVariableOp9^string_lookup_3/None_lookup_table_find/LookupTableFindV2*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 2F
!category_encoding_3/Assert/Assert!category_encoding_3/Assert/Assert2P
&normalization_3/Reshape/ReadVariableOp&normalization_3/Reshape/ReadVariableOp2T
(normalization_3/Reshape_1/ReadVariableOp(normalization_3/Reshape_1/ReadVariableOp2t
8string_lookup_3/None_lookup_table_find/LookupTableFindV28string_lookup_3/None_lookup_table_find/LookupTableFindV2:M I
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
month_sold:WS
'
_output_shapes
:?????????
(
_user_specified_nameoperating_cost:NJ
'
_output_shapes
:?????????

_user_specified_namerooms:UQ
'
_output_shapes
:?????????
&
_user_specified_namesquare_meter:]Y
'
_output_shapes
:?????????
.
_user_specified_nameyear_of_construction:RN
'
_output_shapes
:?????????
#
_user_specified_name	year_sold:	

_output_shapes
: 
?
-
__inference__destroyer_230333
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
?
?
C__inference_model_7_layer_call_and_return_conditional_losses_229338

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
model_6_229311
model_6_229313	
model_6_229315:
model_6_229317:'
sequential_3_229320:
??"
sequential_3_229322:	?&
sequential_3_229324:	?@!
sequential_3_229326:@%
sequential_3_229328:@ !
sequential_3_229330: %
sequential_3_229332: !
sequential_3_229334:
identity??model_6/StatefulPartitionedCall?$sequential_3/StatefulPartitionedCall?
model_6/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7model_6_229311model_6_229313model_6_229315model_6_229317*
Tin
2	*
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
C__inference_model_6_layer_call_and_return_conditional_losses_2287512!
model_6/StatefulPartitionedCall?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall(model_6/StatefulPartitionedCall:output:0sequential_3_229320sequential_3_229322sequential_3_229324sequential_3_229326sequential_3_229328sequential_3_229330sequential_3_229332sequential_3_229334*
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
GPU2*0J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_2290962&
$sequential_3/StatefulPartitionedCall?
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0 ^model_6/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2B
model_6/StatefulPartitionedCallmodel_6/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:O K
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
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:	

_output_shapes
: 
?V
?
C__inference_model_6_layer_call_and_return_conditional_losses_229943
inputs_area
inputs_garden_area
inputs_month_sold
inputs_operating_cost
inputs_rooms
inputs_square_meter
inputs_year_of_construction
inputs_year_soldI
Estring_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value	=
/normalization_3_reshape_readvariableop_resource:?
1normalization_3_reshape_1_readvariableop_resource:
identity??!category_encoding_3/Assert/Assert?&normalization_3/Reshape/ReadVariableOp?(normalization_3/Reshape_1/ReadVariableOp?8string_lookup_3/None_lookup_table_find/LookupTableFindV2?
8string_lookup_3/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleinputs_areaFstring_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_3/None_lookup_table_find/LookupTableFindV2x
concatenate_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_6/concat/axis?
concatenate_6/concatConcatV2inputs_roomsinputs_square_meterinputs_garden_areainputs_operating_costinputs_year_of_constructioninputs_year_soldinputs_month_sold"concatenate_6/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate_6/concat?
&normalization_3/Reshape/ReadVariableOpReadVariableOp/normalization_3_reshape_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization_3/Reshape/ReadVariableOp?
normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_3/Reshape/shape?
normalization_3/ReshapeReshape.normalization_3/Reshape/ReadVariableOp:value:0&normalization_3/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_3/Reshape?
(normalization_3/Reshape_1/ReadVariableOpReadVariableOp1normalization_3_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02*
(normalization_3/Reshape_1/ReadVariableOp?
normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_3/Reshape_1/shape?
normalization_3/Reshape_1Reshape0normalization_3/Reshape_1/ReadVariableOp:value:0(normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_3/Reshape_1?
normalization_3/subSubconcatenate_6/concat:output:0 normalization_3/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization_3/sub?
normalization_3/SqrtSqrt"normalization_3/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_3/Maximum/y?
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization_3/Maximum?
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization_3/truediv?
category_encoding_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_3/Const?
category_encoding_3/MaxMaxAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0"category_encoding_3/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_3/Max?
category_encoding_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_3/Const_1?
category_encoding_3/MinMinAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0$category_encoding_3/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_3/Min{
category_encoding_3/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding_3/Cast/x?
category_encoding_3/CastCast#category_encoding_3/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_3/Cast?
category_encoding_3/GreaterGreatercategory_encoding_3/Cast:y:0 category_encoding_3/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_3/Greater~
category_encoding_3/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_3/Cast_1/x?
category_encoding_3/Cast_1Cast%category_encoding_3/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_3/Cast_1?
 category_encoding_3/GreaterEqualGreaterEqual category_encoding_3/Min:output:0category_encoding_3/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/GreaterEqual?
category_encoding_3/LogicalAnd
LogicalAndcategory_encoding_3/Greater:z:0$category_encoding_3/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_3/LogicalAnd?
 category_encoding_3/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=3002"
 category_encoding_3/Assert/Const?
(category_encoding_3/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=3002*
(category_encoding_3/Assert/Assert/data_0?
!category_encoding_3/Assert/AssertAssert"category_encoding_3/LogicalAnd:z:01category_encoding_3/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2#
!category_encoding_3/Assert/Assert?
"category_encoding_3/bincount/ShapeShapeAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_3/bincount/Shape?
"category_encoding_3/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_3/bincount/Const?
!category_encoding_3/bincount/ProdProd+category_encoding_3/bincount/Shape:output:0+category_encoding_3/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_3/bincount/Prod?
&category_encoding_3/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_3/bincount/Greater/y?
$category_encoding_3/bincount/GreaterGreater*category_encoding_3/bincount/Prod:output:0/category_encoding_3/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_3/bincount/Greater?
!category_encoding_3/bincount/CastCast(category_encoding_3/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_3/bincount/Cast?
$category_encoding_3/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_3/bincount/Const_1?
 category_encoding_3/bincount/MaxMaxAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_3/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/Max?
"category_encoding_3/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_3/bincount/add/y?
 category_encoding_3/bincount/addAddV2)category_encoding_3/bincount/Max:output:0+category_encoding_3/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/add?
 category_encoding_3/bincount/mulMul%category_encoding_3/bincount/Cast:y:0$category_encoding_3/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/mul?
&category_encoding_3/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_3/bincount/minlength?
$category_encoding_3/bincount/MaximumMaximum/category_encoding_3/bincount/minlength:output:0$category_encoding_3/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_3/bincount/Maximum?
&category_encoding_3/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_3/bincount/maxlength?
$category_encoding_3/bincount/MinimumMinimum/category_encoding_3/bincount/maxlength:output:0(category_encoding_3/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_3/bincount/Minimum?
$category_encoding_3/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_3/bincount/Const_2?
*category_encoding_3/bincount/DenseBincountDenseBincountAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_3/bincount/Minimum:z:0-category_encoding_3/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2,
*category_encoding_3/bincount/DenseBincountx
concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_7/concat/axis?
concatenate_7/concatConcatV2normalization_3/truediv:z:03category_encoding_3/bincount/DenseBincount:output:0"concatenate_7/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate_7/concat?
IdentityIdentityconcatenate_7/concat:output:0"^category_encoding_3/Assert/Assert'^normalization_3/Reshape/ReadVariableOp)^normalization_3/Reshape_1/ReadVariableOp9^string_lookup_3/None_lookup_table_find/LookupTableFindV2*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 2F
!category_encoding_3/Assert/Assert!category_encoding_3/Assert/Assert2P
&normalization_3/Reshape/ReadVariableOp&normalization_3/Reshape/ReadVariableOp2T
(normalization_3/Reshape_1/ReadVariableOp(normalization_3/Reshape_1/ReadVariableOp2t
8string_lookup_3/None_lookup_table_find/LookupTableFindV28string_lookup_3/None_lookup_table_find/LookupTableFindV2:T P
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
_user_specified_nameinputs/month_sold:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/operating_cost:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/rooms:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/square_meter:d`
'
_output_shapes
:?????????
5
_user_specified_nameinputs/year_of_construction:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/year_sold:	

_output_shapes
: 
?
?
D__inference_dense_12_layer_call_and_return_conditional_losses_230252

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
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
2*,
_gradient_op_typeCustomGradient-230245*<
_output_shapes*
(:??????????:??????????2
	IdentityN?

Identity_1IdentityIdentityN:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
R
__inference__creator_230323
identity: ??string_lookup_3_index_table?
string_lookup_3_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_206512*
value_dtype0	2
string_lookup_3_index_table?
IdentityIdentity*string_lookup_3_index_table:table_handle:0^string_lookup_3_index_table*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2:
string_lookup_3_index_tablestring_lookup_3_index_table"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
5
area-
serving_default_area:0?????????
C
garden_area4
serving_default_garden_area:0?????????
A

month_sold3
serving_default_month_sold:0?????????
I
operating_cost7
 serving_default_operating_cost:0?????????
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
sequential_30
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-0
	layer-8

layer_with_weights-1

layer-9
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"??
_tf_keras_network??{"name": "model_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}, "name": "area", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "garden_area"}, "name": "garden_area", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}, "name": "month_sold", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "operating_cost"}, "name": "operating_cost", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}, "name": "rooms", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}, "name": "square_meter", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_of_construction"}, "name": "year_of_construction", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}, "name": "year_sold", "inbound_nodes": [], "shared_object_id": 7}, {"class_name": "Functional", "config": {"name": "model_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}, "name": "rooms", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}, "name": "square_meter", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "garden_area"}, "name": "garden_area", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "operating_cost"}, "name": "operating_cost", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_of_construction"}, "name": "year_of_construction", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}, "name": "year_sold", "inbound_nodes": [], "shared_object_id": 7}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}, "name": "month_sold", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}, "name": "area", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Concatenate", "config": {"name": "concatenate_6", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_6", "inbound_nodes": [[["rooms", 0, 0, {}], ["square_meter", 0, 0, {}], ["garden_area", 0, 0, {}], ["operating_cost", 0, 0, {}], ["year_of_construction", 0, 0, {}], ["year_sold", 0, 0, {}], ["month_sold", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_3", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "output_mode": "int", "pad_to_max_tokens": false, "vocabulary_size": 300, "vocabulary": null, "encoding": "utf-8"}, "name": "string_lookup_3", "inbound_nodes": [[["area", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "name": "normalization_3", "inbound_nodes": [[["concatenate_6", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_3", "trainable": true, "dtype": "float32", "num_tokens": 300, "output_mode": "binary", "sparse": false}, "name": "category_encoding_3", "inbound_nodes": [[["string_lookup_3", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_7", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_7", "inbound_nodes": [[["normalization_3", 0, 0, {}], ["category_encoding_3", 0, 0, {}]]]}], "input_layers": {"area": ["area", 0, 0], "rooms": ["rooms", 0, 0], "square_meter": ["square_meter", 0, 0], "garden_area": ["garden_area", 0, 0], "operating_cost": ["operating_cost", 0, 0], "year_of_construction": ["year_of_construction", 0, 0], "year_sold": ["year_sold", 0, 0], "month_sold": ["month_sold", 0, 0]}, "output_layers": [["concatenate_7", 0, 0]]}, "name": "model_6", "inbound_nodes": [{"area": ["area", 0, 0, {}], "rooms": ["rooms", 0, 0, {}], "square_meter": ["square_meter", 0, 0, {}], "garden_area": ["garden_area", 0, 0, {}], "operating_cost": ["operating_cost", 0, 0, {}], "year_of_construction": ["year_of_construction", 0, 0, {}], "year_sold": ["year_sold", 0, 0, {}], "month_sold": ["month_sold", 0, 0, {}]}]}, {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 307]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_12_input"}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 128, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential_3", "inbound_nodes": [[["model_6", 1, 0, {}]]]}], "input_layers": {"area": ["area", 0, 0], "rooms": ["rooms", 0, 0], "square_meter": ["square_meter", 0, 0], "garden_area": ["garden_area", 0, 0], "operating_cost": ["operating_cost", 0, 0], "year_of_construction": ["year_of_construction", 0, 0], "year_sold": ["year_sold", 0, 0], "month_sold": ["month_sold", 0, 0]}, "output_layers": [["sequential_3", 1, 0]]}, "shared_object_id": 28, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"area": {"class_name": "TensorShape", "items": [null, 1]}, "rooms": {"class_name": "TensorShape", "items": [null, 1]}, "square_meter": {"class_name": "TensorShape", "items": [null, 1]}, "garden_area": {"class_name": "TensorShape", "items": [null, 1]}, "operating_cost": {"class_name": "TensorShape", "items": [null, 1]}, "year_of_construction": {"class_name": "TensorShape", "items": [null, 1]}, "year_sold": {"class_name": "TensorShape", "items": [null, 1]}, "month_sold": {"class_name": "TensorShape", "items": [null, 1]}}, "is_graph_network": true, "save_spec": {"area": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "string", "area"]}, "rooms": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "rooms"]}, "square_meter": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "square_meter"]}, "garden_area": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "garden_area"]}, "operating_cost": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "operating_cost"]}, "year_of_construction": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "year_of_construction"]}, "year_sold": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "year_sold"]}, "month_sold": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "month_sold"]}}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}, "name": "area", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "garden_area"}, "name": "garden_area", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}, "name": "month_sold", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "operating_cost"}, "name": "operating_cost", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}, "name": "rooms", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}, "name": "square_meter", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_of_construction"}, "name": "year_of_construction", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}, "name": "year_sold", "inbound_nodes": [], "shared_object_id": 7}, {"class_name": "Functional", "config": {"name": "model_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}, "name": "rooms", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}, "name": "square_meter", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "garden_area"}, "name": "garden_area", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "operating_cost"}, "name": "operating_cost", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_of_construction"}, "name": "year_of_construction", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}, "name": "year_sold", "inbound_nodes": [], "shared_object_id": 7}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}, "name": "month_sold", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}, "name": "area", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Concatenate", "config": {"name": "concatenate_6", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_6", "inbound_nodes": [[["rooms", 0, 0, {}], ["square_meter", 0, 0, {}], ["garden_area", 0, 0, {}], ["operating_cost", 0, 0, {}], ["year_of_construction", 0, 0, {}], ["year_sold", 0, 0, {}], ["month_sold", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_3", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "output_mode": "int", "pad_to_max_tokens": false, "vocabulary_size": 300, "vocabulary": null, "encoding": "utf-8"}, "name": "string_lookup_3", "inbound_nodes": [[["area", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "name": "normalization_3", "inbound_nodes": [[["concatenate_6", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_3", "trainable": true, "dtype": "float32", "num_tokens": 300, "output_mode": "binary", "sparse": false}, "name": "category_encoding_3", "inbound_nodes": [[["string_lookup_3", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_7", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_7", "inbound_nodes": [[["normalization_3", 0, 0, {}], ["category_encoding_3", 0, 0, {}]]]}], "input_layers": {"area": ["area", 0, 0], "rooms": ["rooms", 0, 0], "square_meter": ["square_meter", 0, 0], "garden_area": ["garden_area", 0, 0], "operating_cost": ["operating_cost", 0, 0], "year_of_construction": ["year_of_construction", 0, 0], "year_sold": ["year_sold", 0, 0], "month_sold": ["month_sold", 0, 0]}, "output_layers": [["concatenate_7", 0, 0]]}, "name": "model_6", "inbound_nodes": [{"area": ["area", 0, 0, {}], "rooms": ["rooms", 0, 0, {}], "square_meter": ["square_meter", 0, 0, {}], "garden_area": ["garden_area", 0, 0, {}], "operating_cost": ["operating_cost", 0, 0, {}], "year_of_construction": ["year_of_construction", 0, 0, {}], "year_sold": ["year_sold", 0, 0, {}], "month_sold": ["month_sold", 0, 0, {}]}], "shared_object_id": 13}, {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 307]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_12_input"}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 128, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential_3", "inbound_nodes": [[["model_6", 1, 0, {}]]], "shared_object_id": 27}], "input_layers": {"area": ["area", 0, 0], "rooms": ["rooms", 0, 0], "square_meter": ["square_meter", 0, 0], "garden_area": ["garden_area", 0, 0], "operating_cost": ["operating_cost", 0, 0], "year_of_construction": ["year_of_construction", 0, 0], "year_sold": ["year_sold", 0, 0], "month_sold": ["month_sold", 0, 0]}, "output_layers": [["sequential_3", 1, 0]]}}, "training_config": {"loss": {"class_name": "MeanAbsoluteError", "config": {"reduction": "auto", "name": "mean_absolute_error"}, "shared_object_id": 37}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "area", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "garden_area", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "garden_area"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "month_sold", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "operating_cost", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "operating_cost"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "rooms", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "square_meter", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "year_of_construction", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_of_construction"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "year_sold", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}}
?a
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
layer-8
layer_with_weights-0
layer-9
layer_with_weights-1
layer-10
layer-11
layer-12
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?^
_tf_keras_network?^{"name": "model_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}, "name": "rooms", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}, "name": "square_meter", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "garden_area"}, "name": "garden_area", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "operating_cost"}, "name": "operating_cost", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_of_construction"}, "name": "year_of_construction", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}, "name": "year_sold", "inbound_nodes": [], "shared_object_id": 7}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}, "name": "month_sold", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}, "name": "area", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Concatenate", "config": {"name": "concatenate_6", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_6", "inbound_nodes": [[["rooms", 0, 0, {}], ["square_meter", 0, 0, {}], ["garden_area", 0, 0, {}], ["operating_cost", 0, 0, {}], ["year_of_construction", 0, 0, {}], ["year_sold", 0, 0, {}], ["month_sold", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_3", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "output_mode": "int", "pad_to_max_tokens": false, "vocabulary_size": 300, "vocabulary": null, "encoding": "utf-8"}, "name": "string_lookup_3", "inbound_nodes": [[["area", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "name": "normalization_3", "inbound_nodes": [[["concatenate_6", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_3", "trainable": true, "dtype": "float32", "num_tokens": 300, "output_mode": "binary", "sparse": false}, "name": "category_encoding_3", "inbound_nodes": [[["string_lookup_3", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_7", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_7", "inbound_nodes": [[["normalization_3", 0, 0, {}], ["category_encoding_3", 0, 0, {}]]]}], "input_layers": {"area": ["area", 0, 0], "rooms": ["rooms", 0, 0], "square_meter": ["square_meter", 0, 0], "garden_area": ["garden_area", 0, 0], "operating_cost": ["operating_cost", 0, 0], "year_of_construction": ["year_of_construction", 0, 0], "year_sold": ["year_sold", 0, 0], "month_sold": ["month_sold", 0, 0]}, "output_layers": [["concatenate_7", 0, 0]]}, "inbound_nodes": [{"area": ["area", 0, 0, {}], "rooms": ["rooms", 0, 0, {}], "square_meter": ["square_meter", 0, 0, {}], "garden_area": ["garden_area", 0, 0, {}], "operating_cost": ["operating_cost", 0, 0, {}], "year_of_construction": ["year_of_construction", 0, 0, {}], "year_sold": ["year_sold", 0, 0, {}], "month_sold": ["month_sold", 0, 0, {}]}], "shared_object_id": 13, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"area": {"class_name": "TensorShape", "items": [null, 1]}, "rooms": {"class_name": "TensorShape", "items": [null, 1]}, "square_meter": {"class_name": "TensorShape", "items": [null, 1]}, "garden_area": {"class_name": "TensorShape", "items": [null, 1]}, "operating_cost": {"class_name": "TensorShape", "items": [null, 1]}, "year_of_construction": {"class_name": "TensorShape", "items": [null, 1]}, "year_sold": {"class_name": "TensorShape", "items": [null, 1]}, "month_sold": {"class_name": "TensorShape", "items": [null, 1]}}, "is_graph_network": true, "save_spec": {"area": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "string", "area"]}, "rooms": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "rooms"]}, "square_meter": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "square_meter"]}, "garden_area": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "garden_area"]}, "operating_cost": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "operating_cost"]}, "year_of_construction": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "year_of_construction"]}, "year_sold": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "year_sold"]}, "month_sold": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "month_sold"]}}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}, "name": "rooms", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}, "name": "square_meter", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "garden_area"}, "name": "garden_area", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "operating_cost"}, "name": "operating_cost", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_of_construction"}, "name": "year_of_construction", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}, "name": "year_sold", "inbound_nodes": [], "shared_object_id": 7}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}, "name": "month_sold", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}, "name": "area", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Concatenate", "config": {"name": "concatenate_6", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_6", "inbound_nodes": [[["rooms", 0, 0, {}], ["square_meter", 0, 0, {}], ["garden_area", 0, 0, {}], ["operating_cost", 0, 0, {}], ["year_of_construction", 0, 0, {}], ["year_sold", 0, 0, {}], ["month_sold", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "StringLookup", "config": {"name": "string_lookup_3", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "output_mode": "int", "pad_to_max_tokens": false, "vocabulary_size": 300, "vocabulary": null, "encoding": "utf-8"}, "name": "string_lookup_3", "inbound_nodes": [[["area", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Normalization", "config": {"name": "normalization_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "name": "normalization_3", "inbound_nodes": [[["concatenate_6", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_3", "trainable": true, "dtype": "float32", "num_tokens": 300, "output_mode": "binary", "sparse": false}, "name": "category_encoding_3", "inbound_nodes": [[["string_lookup_3", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Concatenate", "config": {"name": "concatenate_7", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_7", "inbound_nodes": [[["normalization_3", 0, 0, {}], ["category_encoding_3", 0, 0, {}]]], "shared_object_id": 12}], "input_layers": {"area": ["area", 0, 0], "rooms": ["rooms", 0, 0], "square_meter": ["square_meter", 0, 0], "garden_area": ["garden_area", 0, 0], "operating_cost": ["operating_cost", 0, 0], "year_of_construction": ["year_of_construction", 0, 0], "year_sold": ["year_sold", 0, 0], "month_sold": ["month_sold", 0, 0]}, "output_layers": [["concatenate_7", 0, 0]]}}}
?)
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
regularization_losses
 trainable_variables
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?'
_tf_keras_sequential?'{"name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 307]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_12_input"}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 128, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "inbound_nodes": [[["model_6", 1, 0, {}]]], "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 307}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 307]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 307]}, "float32", "dense_12_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 307]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_12_input"}, "shared_object_id": 14}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 128, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 23}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 26}]}}}
?
"iter

#beta_1

$beta_2
	%decay
&learning_rate*m?+m?,m?-m?.m?/m?0m?1m?*v?+v?,v?-v?.v?/v?0v?1v?"
	optimizer
o
'1
(2
)3
*4
+5
,6
-7
.8
/9
010
111"
trackable_list_wrapper
 "
trackable_list_wrapper
X
*0
+1
,2
-3
.4
/5
06
17"
trackable_list_wrapper
?
2non_trainable_variables
	variables

3layers
regularization_losses
4metrics
5layer_regularization_losses
trainable_variables
6layer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
7	variables
8trainable_variables
9regularization_losses
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "concatenate_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate_6", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["rooms", 0, 0, {}], ["square_meter", 0, 0, {}], ["garden_area", 0, 0, {}], ["operating_cost", 0, 0, {}], ["year_of_construction", 0, 0, {}], ["year_sold", 0, 0, {}], ["month_sold", 0, 0, {}]]], "shared_object_id": 8, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}]}
?
;state_variables

<_table
=	keras_api"?
_tf_keras_layer?{"name": "string_lookup_3", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": null, "stateful": true, "must_restore_from_config": true, "class_name": "StringLookup", "config": {"name": "string_lookup_3", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "output_mode": "int", "pad_to_max_tokens": false, "vocabulary_size": 300, "vocabulary": null, "encoding": "utf-8"}, "inbound_nodes": [[["area", 0, 0, {}]]], "shared_object_id": 9}
?
>
_keep_axis
?_reduce_axis
@_reduce_axis_mask
A_broadcast_shape
'mean
(variance
	)count
B	keras_api
?_adapt_function"?
_tf_keras_layer?{"name": "normalization_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": true, "must_restore_from_config": true, "class_name": "Normalization", "config": {"name": "normalization_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "inbound_nodes": [[["concatenate_6", 0, 0, {}]]], "shared_object_id": 10, "build_input_shape": [null, 7]}
?
C	keras_api"?
_tf_keras_layer?{"name": "category_encoding_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "CategoryEncoding", "config": {"name": "category_encoding_3", "trainable": true, "dtype": "float32", "num_tokens": 300, "output_mode": "binary", "sparse": false}, "inbound_nodes": [[["string_lookup_3", 0, 0, {}]]], "shared_object_id": 11}
?
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "concatenate_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate_7", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["normalization_3", 0, 0, {}], ["category_encoding_3", 0, 0, {}]]], "shared_object_id": 12, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 7]}, {"class_name": "TensorShape", "items": [null, 300]}]}
5
'1
(2
)3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Hnon_trainable_variables
	variables

Ilayers
regularization_losses
Jmetrics
Klayer_regularization_losses
trainable_variables
Llayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

*kernel
+bias
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 128, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 307}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 307]}}
?

,kernel
-bias
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

.kernel
/bias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

0kernel
1bias
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 26, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 49}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
X
*0
+1
,2
-3
.4
/5
06
17"
trackable_list_wrapper
 "
trackable_list_wrapper
X
*0
+1
,2
-3
.4
/5
06
17"
trackable_list_wrapper
?
]non_trainable_variables
	variables

^layers
regularization_losses
_metrics
`layer_regularization_losses
 trainable_variables
alayer_metrics
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
:2mean
:2variance
:	 2count
#:!
??2dense_12/kernel
:?2dense_12/bias
": 	?@2dense_13/kernel
:@2dense_13/bias
!:@ 2dense_14/kernel
: 2dense_14/bias
!: 2dense_15/kernel
:2dense_15/bias
5
'1
(2
)3"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
'
b0"
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
?
7	variables
8trainable_variables

clayers
9regularization_losses
dmetrics
elayer_regularization_losses
fnon_trainable_variables
glayer_metrics
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
D	variables
Etrainable_variables

hlayers
Fregularization_losses
imetrics
jlayer_regularization_losses
knon_trainable_variables
llayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5
'1
(2
)3"
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
8
9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
M	variables
Ntrainable_variables

mlayers
Oregularization_losses
nmetrics
olayer_regularization_losses
pnon_trainable_variables
qlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Q	variables
Rtrainable_variables

rlayers
Sregularization_losses
smetrics
tlayer_regularization_losses
unon_trainable_variables
vlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
U	variables
Vtrainable_variables

wlayers
Wregularization_losses
xmetrics
ylayer_regularization_losses
znon_trainable_variables
{layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Y	variables
Ztrainable_variables

|layers
[regularization_losses
}metrics
~layer_regularization_losses
non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 50}
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
??2Adam/dense_12/kernel/m
!:?2Adam/dense_12/bias/m
':%	?@2Adam/dense_13/kernel/m
 :@2Adam/dense_13/bias/m
&:$@ 2Adam/dense_14/kernel/m
 : 2Adam/dense_14/bias/m
&:$ 2Adam/dense_15/kernel/m
 :2Adam/dense_15/bias/m
(:&
??2Adam/dense_12/kernel/v
!:?2Adam/dense_12/bias/v
':%	?@2Adam/dense_13/kernel/v
 :@2Adam/dense_13/bias/v
&:$@ 2Adam/dense_14/kernel/v
 : 2Adam/dense_14/bias/v
&:$ 2Adam/dense_15/kernel/v
 :2Adam/dense_15/bias/v
?2?
!__inference__wrapped_model_228660?
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
annotations? *???
???
&
area?
area?????????
4
garden_area%?"
garden_area?????????
2

month_sold$?!

month_sold?????????
:
operating_cost(?%
operating_cost?????????
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
C__inference_model_7_layer_call_and_return_conditional_losses_229719
C__inference_model_7_layer_call_and_return_conditional_losses_229810
C__inference_model_7_layer_call_and_return_conditional_losses_229547
C__inference_model_7_layer_call_and_return_conditional_losses_229584?
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
(__inference_model_7_layer_call_fn_229365
(__inference_model_7_layer_call_fn_229846
(__inference_model_7_layer_call_fn_229882
(__inference_model_7_layer_call_fn_229510?
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
C__inference_model_6_layer_call_and_return_conditional_losses_229943
C__inference_model_6_layer_call_and_return_conditional_losses_230004
C__inference_model_6_layer_call_and_return_conditional_losses_228959
C__inference_model_6_layer_call_and_return_conditional_losses_229018?
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
(__inference_model_6_layer_call_fn_228762
(__inference_model_6_layer_call_fn_230024
(__inference_model_6_layer_call_fn_230044
(__inference_model_6_layer_call_fn_228900?
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
H__inference_sequential_3_layer_call_and_return_conditional_losses_230078
H__inference_sequential_3_layer_call_and_return_conditional_losses_230112
H__inference_sequential_3_layer_call_and_return_conditional_losses_229266
H__inference_sequential_3_layer_call_and_return_conditional_losses_229290?
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
?2?
-__inference_sequential_3_layer_call_fn_229115
-__inference_sequential_3_layer_call_fn_230133
-__inference_sequential_3_layer_call_fn_230154
-__inference_sequential_3_layer_call_fn_229242?
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
$__inference_signature_wrapper_229628areagarden_area
month_soldoperating_costroomssquare_meteryear_of_construction	year_sold"?
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
I__inference_concatenate_6_layer_call_and_return_conditional_losses_230166?
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
.__inference_concatenate_6_layer_call_fn_230177?
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
__inference_adapt_step_230223?
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
I__inference_concatenate_7_layer_call_and_return_conditional_losses_230230?
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
.__inference_concatenate_7_layer_call_fn_230236?
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
D__inference_dense_12_layer_call_and_return_conditional_losses_230252?
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
)__inference_dense_12_layer_call_fn_230261?
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
D__inference_dense_13_layer_call_and_return_conditional_losses_230271?
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
)__inference_dense_13_layer_call_fn_230280?
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
D__inference_dense_14_layer_call_and_return_conditional_losses_230290?
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
)__inference_dense_14_layer_call_fn_230299?
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
D__inference_dense_15_layer_call_and_return_conditional_losses_230309?
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
)__inference_dense_15_layer_call_fn_230318?
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
__inference__creator_230323?
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
__inference__initializer_230328?
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
__inference__destroyer_230333?
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
__inference_save_fn_230352checkpoint_key"?
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
__inference_restore_fn_230360restored_tensors_0restored_tensors_1"?
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
Const7
__inference__creator_230323?

? 
? "? 9
__inference__destroyer_230333?

? 
? "? ;
__inference__initializer_230328?

? 
? "? ?
!__inference__wrapped_model_228660?<?'(*+,-./01???
???
???
&
area?
area?????????
4
garden_area%?"
garden_area?????????
2

month_sold$?!

month_sold?????????
:
operating_cost(?%
operating_cost?????????
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
sequential_3&?#
sequential_3?????????m
__inference_adapt_step_230223L)'(A?>
7?4
2?/?
??????????IteratorSpec
? "
 ?
I__inference_concatenate_6_layer_call_and_return_conditional_losses_230166????
???
???
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
"?
inputs/6?????????
? "%?"
?
0?????????
? ?
.__inference_concatenate_6_layer_call_fn_230177????
???
???
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
"?
inputs/6?????????
? "???????????
I__inference_concatenate_7_layer_call_and_return_conditional_losses_230230?[?X
Q?N
L?I
"?
inputs/0?????????
#? 
inputs/1??????????
? "&?#
?
0??????????
? ?
.__inference_concatenate_7_layer_call_fn_230236x[?X
Q?N
L?I
"?
inputs/0?????????
#? 
inputs/1??????????
? "????????????
D__inference_dense_12_layer_call_and_return_conditional_losses_230252^*+0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
)__inference_dense_12_layer_call_fn_230261Q*+0?-
&?#
!?
inputs??????????
? "????????????
D__inference_dense_13_layer_call_and_return_conditional_losses_230271],-0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? }
)__inference_dense_13_layer_call_fn_230280P,-0?-
&?#
!?
inputs??????????
? "??????????@?
D__inference_dense_14_layer_call_and_return_conditional_losses_230290\.//?,
%?"
 ?
inputs?????????@
? "%?"
?
0????????? 
? |
)__inference_dense_14_layer_call_fn_230299O.//?,
%?"
 ?
inputs?????????@
? "?????????? ?
D__inference_dense_15_layer_call_and_return_conditional_losses_230309\01/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? |
)__inference_dense_15_layer_call_fn_230318O01/?,
%?"
 ?
inputs????????? 
? "???????????
C__inference_model_6_layer_call_and_return_conditional_losses_228959?<?'(???
???
???
&
area?
area?????????
4
garden_area%?"
garden_area?????????
2

month_sold$?!

month_sold?????????
:
operating_cost(?%
operating_cost?????????
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
C__inference_model_6_layer_call_and_return_conditional_losses_229018?<?'(???
???
???
&
area?
area?????????
4
garden_area%?"
garden_area?????????
2

month_sold$?!

month_sold?????????
:
operating_cost(?%
operating_cost?????????
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
C__inference_model_6_layer_call_and_return_conditional_losses_229943?<?'(???
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
A
operating_cost/?,
inputs/operating_cost?????????
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
C__inference_model_6_layer_call_and_return_conditional_losses_230004?<?'(???
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
A
operating_cost/?,
inputs/operating_cost?????????
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
? ?
(__inference_model_6_layer_call_fn_228762?<?'(???
???
???
&
area?
area?????????
4
garden_area%?"
garden_area?????????
2

month_sold$?!

month_sold?????????
:
operating_cost(?%
operating_cost?????????
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
? "????????????
(__inference_model_6_layer_call_fn_228900?<?'(???
???
???
&
area?
area?????????
4
garden_area%?"
garden_area?????????
2

month_sold$?!

month_sold?????????
:
operating_cost(?%
operating_cost?????????
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
(__inference_model_6_layer_call_fn_230024?<?'(???
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
A
operating_cost/?,
inputs/operating_cost?????????
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
(__inference_model_6_layer_call_fn_230044?<?'(???
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
A
operating_cost/?,
inputs/operating_cost?????????
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
C__inference_model_7_layer_call_and_return_conditional_losses_229547?<?'(*+,-./01???
???
???
&
area?
area?????????
4
garden_area%?"
garden_area?????????
2

month_sold$?!

month_sold?????????
:
operating_cost(?%
operating_cost?????????
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
C__inference_model_7_layer_call_and_return_conditional_losses_229584?<?'(*+,-./01???
???
???
&
area?
area?????????
4
garden_area%?"
garden_area?????????
2

month_sold$?!

month_sold?????????
:
operating_cost(?%
operating_cost?????????
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
? ?
C__inference_model_7_layer_call_and_return_conditional_losses_229719?<?'(*+,-./01???
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
A
operating_cost/?,
inputs/operating_cost?????????
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
? ?
C__inference_model_7_layer_call_and_return_conditional_losses_229810?<?'(*+,-./01???
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
A
operating_cost/?,
inputs/operating_cost?????????
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
? ?
(__inference_model_7_layer_call_fn_229365?<?'(*+,-./01???
???
???
&
area?
area?????????
4
garden_area%?"
garden_area?????????
2

month_sold$?!

month_sold?????????
:
operating_cost(?%
operating_cost?????????
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
? "???????????
(__inference_model_7_layer_call_fn_229510?<?'(*+,-./01???
???
???
&
area?
area?????????
4
garden_area%?"
garden_area?????????
2

month_sold$?!

month_sold?????????
:
operating_cost(?%
operating_cost?????????
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
(__inference_model_7_layer_call_fn_229846?<?'(*+,-./01???
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
A
operating_cost/?,
inputs/operating_cost?????????
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
(__inference_model_7_layer_call_fn_229882?<?'(*+,-./01???
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
A
operating_cost/?,
inputs/operating_cost?????????
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
? "??????????z
__inference_restore_fn_230360Y<K?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_230352?<&?#
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
H__inference_sequential_3_layer_call_and_return_conditional_losses_229266s*+,-./01@?=
6?3
)?&
dense_12_input??????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_229290s*+,-./01@?=
6?3
)?&
dense_12_input??????????
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_230078k*+,-./018?5
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
H__inference_sequential_3_layer_call_and_return_conditional_losses_230112k*+,-./018?5
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
-__inference_sequential_3_layer_call_fn_229115f*+,-./01@?=
6?3
)?&
dense_12_input??????????
p 

 
? "???????????
-__inference_sequential_3_layer_call_fn_229242f*+,-./01@?=
6?3
)?&
dense_12_input??????????
p

 
? "???????????
-__inference_sequential_3_layer_call_fn_230133^*+,-./018?5
.?+
!?
inputs??????????
p 

 
? "???????????
-__inference_sequential_3_layer_call_fn_230154^*+,-./018?5
.?+
!?
inputs??????????
p

 
? "???????????
$__inference_signature_wrapper_229628?<?'(*+,-./01???
? 
???
&
area?
area?????????
4
garden_area%?"
garden_area?????????
2

month_sold$?!

month_sold?????????
:
operating_cost(?%
operating_cost?????????
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
sequential_3&?#
sequential_3?????????