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
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
??*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:?*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	?@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:@*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:@ *
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
: *
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

: *
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
?
string_lookup_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_278*
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
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
??*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*&
shared_nameAdam/dense_1/kernel/m
?
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	?@*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:@ *
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

: *
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
??*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*&
shared_nameAdam/dense_1/kernel/v
?
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	?@*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:@ *
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

: *
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
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
__inference_<lambda>_48695

NoOpNoOp^PartitionedCall
?
Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2string_lookup_index_table*
Tkeys0*
Tvalues0	*,
_class"
 loc:@string_lookup_index_table*
_output_shapes

::
?;
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*?;
value?;B?: B?:
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
layer-0
layer-1
layer-2
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
HF
VARIABLE_VALUEdense/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUE
dense/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_2/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_2/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_3/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_3/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
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
0
1
2
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
ki
VARIABLE_VALUEAdam/dense/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEAdam/dense/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_1/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense_1/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_2/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense_2/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_3/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_3/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEAdam/dense/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_1/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense_1/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_2/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense_2/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_3/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_3/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
w
serving_default_areaPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
x
serving_default_floorPlaceholder*'
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
w
serving_default_rentPlaceholder*'
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_areaserving_default_floorserving_default_month_soldserving_default_rentserving_default_roomsserving_default_square_meter$serving_default_year_of_constructionserving_default_year_soldstring_lookup_index_tableConstmeanvariancedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
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
GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_47958
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpmean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpHstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2Jstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2:1total/Read/ReadVariableOpcount_1/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOpConst_1*1
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
GPU2*0J 8? *'
f"R 
__inference__traced_save_48834
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratemeanvariancecountdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasstring_lookup_index_tabletotalcount_1Adam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/v*/
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
!__inference__traced_restore_48949??
?
?
B__inference_model_1_layer_call_and_return_conditional_losses_47777

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
model_47750
model_47752	
model_47754:
model_47756:$
sequential_47759:
??
sequential_47761:	?#
sequential_47763:	?@
sequential_47765:@"
sequential_47767:@ 
sequential_47769: "
sequential_47771: 
sequential_47773:
identity??model/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7model_47750model_47752model_47754model_47756*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_471992
model/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0sequential_47759sequential_47761sequential_47763sequential_47765sequential_47767sequential_47769sequential_47771sequential_47773*
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
GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_475322$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^model/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:O K
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
?	
?
*__inference_sequential_layer_call_fn_47445
dense_input
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_474262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:??????????
%
_user_specified_namedense_input
?
?
'__inference_model_1_layer_call_fn_48176
inputs_area
inputs_floor
inputs_month_sold
inputs_rent
inputs_rooms
inputs_square_meter
inputs_year_of_construction
inputs_year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_areainputs_floorinputs_month_soldinputs_rentinputs_roomsinputs_square_meterinputs_year_of_constructioninputs_year_soldunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2*0J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_476682
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
_user_specified_nameinputs/area:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/floor:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/month_sold:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/rent:UQ
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
?'
?
E__inference_sequential_layer_call_and_return_conditional_losses_48442

inputs8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@ 5
'dense_2_biasadd_readvariableop_resource: 8
&dense_3_matmul_readvariableop_resource: 5
'dense_3_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddt
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense/Sigmoid{
	dense/mulMuldense/BiasAdd:output:0dense/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
	dense/muln
dense/IdentityIdentitydense/mul:z:0*
T0*(
_output_shapes
:??????????2
dense/Identity?
dense/IdentityN	IdentityNdense/mul:z:0dense/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-48417*<
_output_shapes*
(:??????????:??????????2
dense/IdentityN?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/IdentityN:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/BiasAdd?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/BiasAdd:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_2/BiasAdd?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/BiasAdd:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAdd?
IdentityIdentitydense_3/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
B__inference_model_1_layer_call_and_return_conditional_losses_47668

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
model_47641
model_47643	
model_47645:
model_47647:$
sequential_47650:
??
sequential_47652:	?#
sequential_47654:	?@
sequential_47656:@"
sequential_47658:@ 
sequential_47660: "
sequential_47662: 
sequential_47664:
identity??model/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7model_47641model_47643model_47645model_47647*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_470812
model/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0sequential_47650sequential_47652sequential_47654sequential_47656sequential_47658sequential_47660sequential_47662sequential_47664*
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
GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_474262$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^model/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:O K
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
?
?
@__inference_dense_layer_call_and_return_conditional_losses_47371

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
_gradient_op_typeCustomGradient-47364*<
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_48212
inputs_area
inputs_floor
inputs_month_sold
inputs_rent
inputs_rooms
inputs_square_meter
inputs_year_of_construction
inputs_year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_areainputs_floorinputs_month_soldinputs_rentinputs_roomsinputs_square_meterinputs_year_of_constructioninputs_year_soldunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2*0J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_477772
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
_user_specified_nameinputs/area:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/floor:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/month_sold:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/rent:UQ
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
?
'__inference_model_1_layer_call_fn_47840
area	
floor

month_sold
rent	
rooms
square_meter
year_of_construction
	year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallareafloor
month_soldrentroomssquare_meteryear_of_construction	year_soldunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2*0J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_477772
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

_user_specified_namearea:NJ
'
_output_shapes
:?????????

_user_specified_namefloor:SO
'
_output_shapes
:?????????
$
_user_specified_name
month_sold:MI
'
_output_shapes
:?????????

_user_specified_namerent:NJ
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
B__inference_dense_3_layer_call_and_return_conditional_losses_47419

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
?S
?
@__inference_model_layer_call_and_return_conditional_losses_47081

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7G
Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handleH
Dstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value	;
-normalization_reshape_readvariableop_resource:=
/normalization_reshape_1_readvariableop_resource:
identity??category_encoding/Assert/Assert?$normalization/Reshape/ReadVariableOp?&normalization/Reshape_1/ReadVariableOp?6string_lookup/None_lookup_table_find/LookupTableFindV2?
6string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handleinputsDstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????28
6string_lookup/None_lookup_table_find/LookupTableFindV2?
concatenate/PartitionedCallPartitionedCallinputs_1inputs_4inputs_5inputs_3inputs_6inputs_7inputs_2*
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
GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_470262
concatenate/PartitionedCall?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape_1?
normalization/subSub$concatenate/PartitionedCall:output:0normalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization/sub{
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization/truediv?
category_encoding/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding/Const?
category_encoding/MaxMax?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0 category_encoding/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding/Max?
category_encoding/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding/Const_1?
category_encoding/MinMin?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0"category_encoding/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding/Minw
category_encoding/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding/Cast/x?
category_encoding/CastCast!category_encoding/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding/Cast?
category_encoding/GreaterGreatercategory_encoding/Cast:y:0category_encoding/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding/Greaterz
category_encoding/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding/Cast_1/x?
category_encoding/Cast_1Cast#category_encoding/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding/Cast_1?
category_encoding/GreaterEqualGreaterEqualcategory_encoding/Min:output:0category_encoding/Cast_1:y:0*
T0	*
_output_shapes
: 2 
category_encoding/GreaterEqual?
category_encoding/LogicalAnd
LogicalAndcategory_encoding/Greater:z:0"category_encoding/GreaterEqual:z:0*
_output_shapes
: 2
category_encoding/LogicalAnd?
category_encoding/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=1812 
category_encoding/Assert/Const?
&category_encoding/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=1812(
&category_encoding/Assert/Assert/data_0?
category_encoding/Assert/AssertAssert category_encoding/LogicalAnd:z:0/category_encoding/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2!
category_encoding/Assert/Assert?
 category_encoding/bincount/ShapeShape?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2"
 category_encoding/bincount/Shape?
 category_encoding/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 category_encoding/bincount/Const?
category_encoding/bincount/ProdProd)category_encoding/bincount/Shape:output:0)category_encoding/bincount/Const:output:0*
T0*
_output_shapes
: 2!
category_encoding/bincount/Prod?
$category_encoding/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2&
$category_encoding/bincount/Greater/y?
"category_encoding/bincount/GreaterGreater(category_encoding/bincount/Prod:output:0-category_encoding/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2$
"category_encoding/bincount/Greater?
category_encoding/bincount/CastCast&category_encoding/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2!
category_encoding/bincount/Cast?
"category_encoding/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"category_encoding/bincount/Const_1?
category_encoding/bincount/MaxMax?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0+category_encoding/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/Max?
 category_encoding/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2"
 category_encoding/bincount/add/y?
category_encoding/bincount/addAddV2'category_encoding/bincount/Max:output:0)category_encoding/bincount/add/y:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/add?
category_encoding/bincount/mulMul#category_encoding/bincount/Cast:y:0"category_encoding/bincount/add:z:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/mul?
$category_encoding/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2&
$category_encoding/bincount/minlength?
"category_encoding/bincount/MaximumMaximum-category_encoding/bincount/minlength:output:0"category_encoding/bincount/mul:z:0*
T0	*
_output_shapes
: 2$
"category_encoding/bincount/Maximum?
$category_encoding/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2&
$category_encoding/bincount/maxlength?
"category_encoding/bincount/MinimumMinimum-category_encoding/bincount/maxlength:output:0&category_encoding/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2$
"category_encoding/bincount/Minimum?
"category_encoding/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2$
"category_encoding/bincount/Const_2?
(category_encoding/bincount/DenseBincountDenseBincount?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0&category_encoding/bincount/Minimum:z:0+category_encoding/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2*
(category_encoding/bincount/DenseBincount?
concatenate_1/PartitionedCallPartitionedCallnormalization/truediv:z:01category_encoding/bincount/DenseBincount:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_470782
concatenate_1/PartitionedCall?
IdentityIdentity&concatenate_1/PartitionedCall:output:0 ^category_encoding/Assert/Assert%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp7^string_lookup/None_lookup_table_find/LookupTableFindV2*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 2B
category_encoding/Assert/Assertcategory_encoding/Assert/Assert2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp2p
6string_lookup/None_lookup_table_find/LookupTableFindV26string_lookup/None_lookup_table_find/LookupTableFindV2:O K
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
?	
?
*__inference_sequential_layer_call_fn_48484

inputs
unknown:
??
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
GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_475322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_dense_1_layer_call_fn_48610

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
GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_473872
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
?
?
B__inference_model_1_layer_call_and_return_conditional_losses_47877
area	
floor

month_sold
rent	
rooms
square_meter
year_of_construction
	year_sold
model_47850
model_47852	
model_47854:
model_47856:$
sequential_47859:
??
sequential_47861:	?#
sequential_47863:	?@
sequential_47865:@"
sequential_47867:@ 
sequential_47869: "
sequential_47871: 
sequential_47873:
identity??model/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCallareafloor
month_soldrentroomssquare_meteryear_of_construction	year_soldmodel_47850model_47852model_47854model_47856*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_470812
model/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0sequential_47859sequential_47861sequential_47863sequential_47865sequential_47867sequential_47869sequential_47871sequential_47873*
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
GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_474262$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^model/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:M I
'
_output_shapes
:?????????

_user_specified_namearea:NJ
'
_output_shapes
:?????????

_user_specified_namefloor:SO
'
_output_shapes
:?????????
$
_user_specified_name
month_sold:MI
'
_output_shapes
:?????????

_user_specified_namerent:NJ
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
B__inference_dense_3_layer_call_and_return_conditional_losses_48639

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
?
?
%__inference_dense_layer_call_fn_48591

inputs
unknown:
??
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
GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_473712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

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
?I
?
__inference__traced_save_48834
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableopS
Osavev2_string_lookup_index_table_lookup_table_export_values_lookuptableexportv2U
Qsavev2_string_lookup_index_table_lookup_table_export_values_lookuptableexportv2_1	$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopsavev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableopOsavev2_string_lookup_index_table_lookup_table_export_values_lookuptableexportv2Qsavev2_string_lookup_index_table_lookup_table_export_values_lookuptableexportv2_1 savev2_total_read_readvariableop"savev2_count_1_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableopsavev2_const_1"/device:CPU:0*
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
??:?:	?@:@:@ : : :::: : :
??:?:	?@:@:@ : : ::
??:?:	?@:@:@ : : :: 2(
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
??:!
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
??:!
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
??:!
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
F__inference_concatenate_layer_call_and_return_conditional_losses_47026

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
?S
?
@__inference_model_layer_call_and_return_conditional_losses_47348
area	
floor

month_sold
rent	
rooms
square_meter
year_of_construction
	year_soldG
Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handleH
Dstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value	;
-normalization_reshape_readvariableop_resource:=
/normalization_reshape_1_readvariableop_resource:
identity??category_encoding/Assert/Assert?$normalization/Reshape/ReadVariableOp?&normalization/Reshape_1/ReadVariableOp?6string_lookup/None_lookup_table_find/LookupTableFindV2?
6string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handleareaDstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????28
6string_lookup/None_lookup_table_find/LookupTableFindV2?
concatenate/PartitionedCallPartitionedCallfloorroomssquare_meterrentyear_of_construction	year_sold
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
GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_470262
concatenate/PartitionedCall?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape_1?
normalization/subSub$concatenate/PartitionedCall:output:0normalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization/sub{
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization/truediv?
category_encoding/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding/Const?
category_encoding/MaxMax?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0 category_encoding/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding/Max?
category_encoding/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding/Const_1?
category_encoding/MinMin?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0"category_encoding/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding/Minw
category_encoding/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding/Cast/x?
category_encoding/CastCast!category_encoding/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding/Cast?
category_encoding/GreaterGreatercategory_encoding/Cast:y:0category_encoding/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding/Greaterz
category_encoding/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding/Cast_1/x?
category_encoding/Cast_1Cast#category_encoding/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding/Cast_1?
category_encoding/GreaterEqualGreaterEqualcategory_encoding/Min:output:0category_encoding/Cast_1:y:0*
T0	*
_output_shapes
: 2 
category_encoding/GreaterEqual?
category_encoding/LogicalAnd
LogicalAndcategory_encoding/Greater:z:0"category_encoding/GreaterEqual:z:0*
_output_shapes
: 2
category_encoding/LogicalAnd?
category_encoding/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=1812 
category_encoding/Assert/Const?
&category_encoding/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=1812(
&category_encoding/Assert/Assert/data_0?
category_encoding/Assert/AssertAssert category_encoding/LogicalAnd:z:0/category_encoding/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2!
category_encoding/Assert/Assert?
 category_encoding/bincount/ShapeShape?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2"
 category_encoding/bincount/Shape?
 category_encoding/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 category_encoding/bincount/Const?
category_encoding/bincount/ProdProd)category_encoding/bincount/Shape:output:0)category_encoding/bincount/Const:output:0*
T0*
_output_shapes
: 2!
category_encoding/bincount/Prod?
$category_encoding/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2&
$category_encoding/bincount/Greater/y?
"category_encoding/bincount/GreaterGreater(category_encoding/bincount/Prod:output:0-category_encoding/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2$
"category_encoding/bincount/Greater?
category_encoding/bincount/CastCast&category_encoding/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2!
category_encoding/bincount/Cast?
"category_encoding/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"category_encoding/bincount/Const_1?
category_encoding/bincount/MaxMax?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0+category_encoding/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/Max?
 category_encoding/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2"
 category_encoding/bincount/add/y?
category_encoding/bincount/addAddV2'category_encoding/bincount/Max:output:0)category_encoding/bincount/add/y:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/add?
category_encoding/bincount/mulMul#category_encoding/bincount/Cast:y:0"category_encoding/bincount/add:z:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/mul?
$category_encoding/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2&
$category_encoding/bincount/minlength?
"category_encoding/bincount/MaximumMaximum-category_encoding/bincount/minlength:output:0"category_encoding/bincount/mul:z:0*
T0	*
_output_shapes
: 2$
"category_encoding/bincount/Maximum?
$category_encoding/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2&
$category_encoding/bincount/maxlength?
"category_encoding/bincount/MinimumMinimum-category_encoding/bincount/maxlength:output:0&category_encoding/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2$
"category_encoding/bincount/Minimum?
"category_encoding/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2$
"category_encoding/bincount/Const_2?
(category_encoding/bincount/DenseBincountDenseBincount?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0&category_encoding/bincount/Minimum:z:0+category_encoding/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2*
(category_encoding/bincount/DenseBincount?
concatenate_1/PartitionedCallPartitionedCallnormalization/truediv:z:01category_encoding/bincount/DenseBincount:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_470782
concatenate_1/PartitionedCall?
IdentityIdentity&concatenate_1/PartitionedCall:output:0 ^category_encoding/Assert/Assert%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp7^string_lookup/None_lookup_table_find/LookupTableFindV2*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 2B
category_encoding/Assert/Assertcategory_encoding/Assert/Assert2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp2p
6string_lookup/None_lookup_table_find/LookupTableFindV26string_lookup/None_lookup_table_find/LookupTableFindV2:M I
'
_output_shapes
:?????????

_user_specified_namearea:NJ
'
_output_shapes
:?????????

_user_specified_namefloor:SO
'
_output_shapes
:?????????
$
_user_specified_name
month_sold:MI
'
_output_shapes
:?????????

_user_specified_namerent:NJ
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
?S
?
@__inference_model_layer_call_and_return_conditional_losses_47199

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7G
Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handleH
Dstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value	;
-normalization_reshape_readvariableop_resource:=
/normalization_reshape_1_readvariableop_resource:
identity??category_encoding/Assert/Assert?$normalization/Reshape/ReadVariableOp?&normalization/Reshape_1/ReadVariableOp?6string_lookup/None_lookup_table_find/LookupTableFindV2?
6string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handleinputsDstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????28
6string_lookup/None_lookup_table_find/LookupTableFindV2?
concatenate/PartitionedCallPartitionedCallinputs_1inputs_4inputs_5inputs_3inputs_6inputs_7inputs_2*
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
GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_470262
concatenate/PartitionedCall?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape_1?
normalization/subSub$concatenate/PartitionedCall:output:0normalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization/sub{
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization/truediv?
category_encoding/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding/Const?
category_encoding/MaxMax?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0 category_encoding/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding/Max?
category_encoding/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding/Const_1?
category_encoding/MinMin?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0"category_encoding/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding/Minw
category_encoding/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding/Cast/x?
category_encoding/CastCast!category_encoding/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding/Cast?
category_encoding/GreaterGreatercategory_encoding/Cast:y:0category_encoding/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding/Greaterz
category_encoding/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding/Cast_1/x?
category_encoding/Cast_1Cast#category_encoding/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding/Cast_1?
category_encoding/GreaterEqualGreaterEqualcategory_encoding/Min:output:0category_encoding/Cast_1:y:0*
T0	*
_output_shapes
: 2 
category_encoding/GreaterEqual?
category_encoding/LogicalAnd
LogicalAndcategory_encoding/Greater:z:0"category_encoding/GreaterEqual:z:0*
_output_shapes
: 2
category_encoding/LogicalAnd?
category_encoding/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=1812 
category_encoding/Assert/Const?
&category_encoding/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=1812(
&category_encoding/Assert/Assert/data_0?
category_encoding/Assert/AssertAssert category_encoding/LogicalAnd:z:0/category_encoding/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2!
category_encoding/Assert/Assert?
 category_encoding/bincount/ShapeShape?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2"
 category_encoding/bincount/Shape?
 category_encoding/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 category_encoding/bincount/Const?
category_encoding/bincount/ProdProd)category_encoding/bincount/Shape:output:0)category_encoding/bincount/Const:output:0*
T0*
_output_shapes
: 2!
category_encoding/bincount/Prod?
$category_encoding/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2&
$category_encoding/bincount/Greater/y?
"category_encoding/bincount/GreaterGreater(category_encoding/bincount/Prod:output:0-category_encoding/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2$
"category_encoding/bincount/Greater?
category_encoding/bincount/CastCast&category_encoding/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2!
category_encoding/bincount/Cast?
"category_encoding/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"category_encoding/bincount/Const_1?
category_encoding/bincount/MaxMax?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0+category_encoding/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/Max?
 category_encoding/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2"
 category_encoding/bincount/add/y?
category_encoding/bincount/addAddV2'category_encoding/bincount/Max:output:0)category_encoding/bincount/add/y:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/add?
category_encoding/bincount/mulMul#category_encoding/bincount/Cast:y:0"category_encoding/bincount/add:z:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/mul?
$category_encoding/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2&
$category_encoding/bincount/minlength?
"category_encoding/bincount/MaximumMaximum-category_encoding/bincount/minlength:output:0"category_encoding/bincount/mul:z:0*
T0	*
_output_shapes
: 2$
"category_encoding/bincount/Maximum?
$category_encoding/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2&
$category_encoding/bincount/maxlength?
"category_encoding/bincount/MinimumMinimum-category_encoding/bincount/maxlength:output:0&category_encoding/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2$
"category_encoding/bincount/Minimum?
"category_encoding/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2$
"category_encoding/bincount/Const_2?
(category_encoding/bincount/DenseBincountDenseBincount?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0&category_encoding/bincount/Minimum:z:0+category_encoding/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2*
(category_encoding/bincount/DenseBincount?
concatenate_1/PartitionedCallPartitionedCallnormalization/truediv:z:01category_encoding/bincount/DenseBincount:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_470782
concatenate_1/PartitionedCall?
IdentityIdentity&concatenate_1/PartitionedCall:output:0 ^category_encoding/Assert/Assert%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp7^string_lookup/None_lookup_table_find/LookupTableFindV2*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 2B
category_encoding/Assert/Assertcategory_encoding/Assert/Assert2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp2p
6string_lookup/None_lookup_table_find/LookupTableFindV26string_lookup/None_lookup_table_find/LookupTableFindV2:O K
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
?
?
%__inference_model_layer_call_fn_48354
inputs_area
inputs_floor
inputs_month_sold
inputs_rent
inputs_rooms
inputs_square_meter
inputs_year_of_construction
inputs_year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_areainputs_floorinputs_month_soldinputs_rentinputs_roomsinputs_square_meterinputs_year_of_constructioninputs_year_soldunknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_470812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

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
_user_specified_nameinputs/area:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/floor:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/month_sold:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/rent:UQ
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
??
?
!__inference__traced_restore_48949
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: %
assignvariableop_5_mean:)
assignvariableop_6_variance:"
assignvariableop_7_count:	 3
assignvariableop_8_dense_kernel:
??,
assignvariableop_9_dense_bias:	?5
"assignvariableop_10_dense_1_kernel:	?@.
 assignvariableop_11_dense_1_bias:@4
"assignvariableop_12_dense_2_kernel:@ .
 assignvariableop_13_dense_2_bias: 4
"assignvariableop_14_dense_3_kernel: .
 assignvariableop_15_dense_3_bias:_
Ustring_lookup_index_table_table_restore_lookuptableimportv2_string_lookup_index_table: #
assignvariableop_16_total: %
assignvariableop_17_count_1: ;
'assignvariableop_18_adam_dense_kernel_m:
??4
%assignvariableop_19_adam_dense_bias_m:	?<
)assignvariableop_20_adam_dense_1_kernel_m:	?@5
'assignvariableop_21_adam_dense_1_bias_m:@;
)assignvariableop_22_adam_dense_2_kernel_m:@ 5
'assignvariableop_23_adam_dense_2_bias_m: ;
)assignvariableop_24_adam_dense_3_kernel_m: 5
'assignvariableop_25_adam_dense_3_bias_m:;
'assignvariableop_26_adam_dense_kernel_v:
??4
%assignvariableop_27_adam_dense_bias_v:	?<
)assignvariableop_28_adam_dense_1_kernel_v:	?@5
'assignvariableop_29_adam_dense_1_bias_v:@;
)assignvariableop_30_adam_dense_2_kernel_v:@ 5
'assignvariableop_31_adam_dense_2_bias_v: ;
)assignvariableop_32_adam_dense_3_kernel_v: 5
'assignvariableop_33_adam_dense_3_bias_v:
identity_35??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?;string_lookup_index_table_table_restore/LookupTableImportV2?
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
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_3_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_3_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15?
;string_lookup_index_table_table_restore/LookupTableImportV2LookupTableImportV2Ustring_lookup_index_table_table_restore_lookuptableimportv2_string_lookup_index_tableRestoreV2:tensors:16RestoreV2:tensors:17*	
Tin0*

Tout0	*,
_class"
 loc:@string_lookup_index_table*
_output_shapes
 2=
;string_lookup_index_table_table_restore/LookupTableImportV2n
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
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp%assignvariableop_19_adam_dense_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_1_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_1_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_2_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_2_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_3_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_dense_3_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp%assignvariableop_27_adam_dense_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_1_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_1_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_2_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_dense_2_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_3_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp'assignvariableop_33_adam_dense_3_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_339
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp<^string_lookup_index_table_table_restore/LookupTableImportV2"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_34?
Identity_35IdentityIdentity_34:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9<^string_lookup_index_table_table_restore/LookupTableImportV2*
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
AssignVariableOp_9AssignVariableOp_92z
;string_lookup_index_table_table_restore/LookupTableImportV2;string_lookup_index_table_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:2.
,
_class"
 loc:@string_lookup_index_table
?
?
%__inference_model_layer_call_fn_47230
area	
floor

month_sold
rent	
rooms
square_meter
year_of_construction
	year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallareafloor
month_soldrentroomssquare_meteryear_of_construction	year_soldunknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_471992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
'
_output_shapes
:?????????

_user_specified_namearea:NJ
'
_output_shapes
:?????????

_user_specified_namefloor:SO
'
_output_shapes
:?????????
$
_user_specified_name
month_sold:MI
'
_output_shapes
:?????????

_user_specified_namerent:NJ
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
?
?
%__inference_model_layer_call_fn_48374
inputs_area
inputs_floor
inputs_month_sold
inputs_rent
inputs_rooms
inputs_square_meter
inputs_year_of_construction
inputs_year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_areainputs_floorinputs_month_soldinputs_rentinputs_roomsinputs_square_meterinputs_year_of_constructioninputs_year_soldunknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_471992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

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
_user_specified_nameinputs/area:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/floor:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/month_sold:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/rent:UQ
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
B__inference_dense_2_layer_call_and_return_conditional_losses_47403

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
'__inference_dense_3_layer_call_fn_48648

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
GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_474192
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
?
E__inference_sequential_layer_call_and_return_conditional_losses_47620
dense_input
dense_47599:
??
dense_47601:	? 
dense_1_47604:	?@
dense_1_47606:@
dense_2_47609:@ 
dense_2_47611: 
dense_3_47614: 
dense_3_47616:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_47599dense_47601*
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
GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_473712
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_47604dense_1_47606*
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
GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_473872!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_47609dense_2_47611*
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
GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_474032!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_47614dense_3_47616*
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
GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_474192!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:U Q
(
_output_shapes
:??????????
%
_user_specified_namedense_input
?
*
__inference_<lambda>_48695
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
?
?
'__inference_model_1_layer_call_fn_47695
area	
floor

month_sold
rent	
rooms
square_meter
year_of_construction
	year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallareafloor
month_soldrentroomssquare_meteryear_of_construction	year_soldunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2*0J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_476682
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

_user_specified_namearea:NJ
'
_output_shapes
:?????????

_user_specified_namefloor:SO
'
_output_shapes
:?????????
$
_user_specified_name
month_sold:MI
'
_output_shapes
:?????????

_user_specified_namerent:NJ
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
t
H__inference_concatenate_1_layer_call_and_return_conditional_losses_48560
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
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????:??????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?
O
__inference__creator_48653
identity: ??string_lookup_index_table?
string_lookup_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_278*
value_dtype0	2
string_lookup_index_table?
IdentityIdentity(string_lookup_index_table:table_handle:0^string_lookup_index_table*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 26
string_lookup_index_tablestring_lookup_index_table
?	
?
B__inference_dense_1_layer_call_and_return_conditional_losses_47387

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
?'
?
E__inference_sequential_layer_call_and_return_conditional_losses_48408

inputs8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@ 5
'dense_2_biasadd_readvariableop_resource: 8
&dense_3_matmul_readvariableop_resource: 5
'dense_3_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddt
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense/Sigmoid{
	dense/mulMuldense/BiasAdd:output:0dense/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
	dense/muln
dense/IdentityIdentitydense/mul:z:0*
T0*(
_output_shapes
:??????????2
dense/Identity?
dense/IdentityN	IdentityNdense/mul:z:0dense/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-48383*<
_output_shapes*
(:??????????:??????????2
dense/IdentityN?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/IdentityN:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/BiasAdd?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/BiasAdd:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_2/BiasAdd?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/BiasAdd:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAdd?
IdentityIdentitydense_3/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
r
H__inference_concatenate_1_layer_call_and_return_conditional_losses_47078

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
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????:??????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
*__inference_sequential_layer_call_fn_48463

inputs
unknown:
??
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
GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_474262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_save_fn_48682
checkpoint_keyY
Ustring_lookup_index_table_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	??Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2?
Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Ustring_lookup_index_table_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::2J
Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2T
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
IdentityIdentityadd:z:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
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

Identity_1IdentityConst:output:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityOstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2:keys:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
:2

Identity_2?

Identity_3Identity	add_1:z:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
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

Identity_4IdentityConst_1:output:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentityQstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2:values:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
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
Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
?
'__inference_dense_2_layer_call_fn_48629

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
GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_474032
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
??
?
B__inference_model_1_layer_call_and_return_conditional_losses_48049
inputs_area
inputs_floor
inputs_month_sold
inputs_rent
inputs_rooms
inputs_square_meter
inputs_year_of_construction
inputs_year_soldM
Imodel_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handleN
Jmodel_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value	A
3model_normalization_reshape_readvariableop_resource:C
5model_normalization_reshape_1_readvariableop_resource:C
/sequential_dense_matmul_readvariableop_resource:
???
0sequential_dense_biasadd_readvariableop_resource:	?D
1sequential_dense_1_matmul_readvariableop_resource:	?@@
2sequential_dense_1_biasadd_readvariableop_resource:@C
1sequential_dense_2_matmul_readvariableop_resource:@ @
2sequential_dense_2_biasadd_readvariableop_resource: C
1sequential_dense_3_matmul_readvariableop_resource: @
2sequential_dense_3_biasadd_readvariableop_resource:
identity??%model/category_encoding/Assert/Assert?*model/normalization/Reshape/ReadVariableOp?,model/normalization/Reshape_1/ReadVariableOp?<model/string_lookup/None_lookup_table_find/LookupTableFindV2?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?)sequential/dense_2/BiasAdd/ReadVariableOp?(sequential/dense_2/MatMul/ReadVariableOp?)sequential/dense_3/BiasAdd/ReadVariableOp?(sequential/dense_3/MatMul/ReadVariableOp?
<model/string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Imodel_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handleinputs_areaJmodel_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2>
<model/string_lookup/None_lookup_table_find/LookupTableFindV2?
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis?
model/concatenate/concatConcatV2inputs_floorinputs_roomsinputs_square_meterinputs_rentinputs_year_of_constructioninputs_year_soldinputs_month_sold&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
model/concatenate/concat?
*model/normalization/Reshape/ReadVariableOpReadVariableOp3model_normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02,
*model/normalization/Reshape/ReadVariableOp?
!model/normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!model/normalization/Reshape/shape?
model/normalization/ReshapeReshape2model/normalization/Reshape/ReadVariableOp:value:0*model/normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
model/normalization/Reshape?
,model/normalization/Reshape_1/ReadVariableOpReadVariableOp5model_normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,model/normalization/Reshape_1/ReadVariableOp?
#model/normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2%
#model/normalization/Reshape_1/shape?
model/normalization/Reshape_1Reshape4model/normalization/Reshape_1/ReadVariableOp:value:0,model/normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
model/normalization/Reshape_1?
model/normalization/subSub!model/concatenate/concat:output:0$model/normalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2
model/normalization/sub?
model/normalization/SqrtSqrt&model/normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
model/normalization/Sqrt?
model/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
model/normalization/Maximum/y?
model/normalization/MaximumMaximummodel/normalization/Sqrt:y:0&model/normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
model/normalization/Maximum?
model/normalization/truedivRealDivmodel/normalization/sub:z:0model/normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????2
model/normalization/truediv?
model/category_encoding/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
model/category_encoding/Const?
model/category_encoding/MaxMaxEmodel/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0&model/category_encoding/Const:output:0*
T0	*
_output_shapes
: 2
model/category_encoding/Max?
model/category_encoding/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
model/category_encoding/Const_1?
model/category_encoding/MinMinEmodel/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0(model/category_encoding/Const_1:output:0*
T0	*
_output_shapes
: 2
model/category_encoding/Min?
model/category_encoding/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2 
model/category_encoding/Cast/x?
model/category_encoding/CastCast'model/category_encoding/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
model/category_encoding/Cast?
model/category_encoding/GreaterGreater model/category_encoding/Cast:y:0$model/category_encoding/Max:output:0*
T0	*
_output_shapes
: 2!
model/category_encoding/Greater?
 model/category_encoding/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2"
 model/category_encoding/Cast_1/x?
model/category_encoding/Cast_1Cast)model/category_encoding/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2 
model/category_encoding/Cast_1?
$model/category_encoding/GreaterEqualGreaterEqual$model/category_encoding/Min:output:0"model/category_encoding/Cast_1:y:0*
T0	*
_output_shapes
: 2&
$model/category_encoding/GreaterEqual?
"model/category_encoding/LogicalAnd
LogicalAnd#model/category_encoding/Greater:z:0(model/category_encoding/GreaterEqual:z:0*
_output_shapes
: 2$
"model/category_encoding/LogicalAnd?
$model/category_encoding/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=1812&
$model/category_encoding/Assert/Const?
,model/category_encoding/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=1812.
,model/category_encoding/Assert/Assert/data_0?
%model/category_encoding/Assert/AssertAssert&model/category_encoding/LogicalAnd:z:05model/category_encoding/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2'
%model/category_encoding/Assert/Assert?
&model/category_encoding/bincount/ShapeShapeEmodel/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2(
&model/category_encoding/bincount/Shape?
&model/category_encoding/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model/category_encoding/bincount/Const?
%model/category_encoding/bincount/ProdProd/model/category_encoding/bincount/Shape:output:0/model/category_encoding/bincount/Const:output:0*
T0*
_output_shapes
: 2'
%model/category_encoding/bincount/Prod?
*model/category_encoding/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2,
*model/category_encoding/bincount/Greater/y?
(model/category_encoding/bincount/GreaterGreater.model/category_encoding/bincount/Prod:output:03model/category_encoding/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2*
(model/category_encoding/bincount/Greater?
%model/category_encoding/bincount/CastCast,model/category_encoding/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2'
%model/category_encoding/bincount/Cast?
(model/category_encoding/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2*
(model/category_encoding/bincount/Const_1?
$model/category_encoding/bincount/MaxMaxEmodel/string_lookup/None_lookup_table_find/LookupTableFindV2:values:01model/category_encoding/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2&
$model/category_encoding/bincount/Max?
&model/category_encoding/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&model/category_encoding/bincount/add/y?
$model/category_encoding/bincount/addAddV2-model/category_encoding/bincount/Max:output:0/model/category_encoding/bincount/add/y:output:0*
T0	*
_output_shapes
: 2&
$model/category_encoding/bincount/add?
$model/category_encoding/bincount/mulMul)model/category_encoding/bincount/Cast:y:0(model/category_encoding/bincount/add:z:0*
T0	*
_output_shapes
: 2&
$model/category_encoding/bincount/mul?
*model/category_encoding/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2,
*model/category_encoding/bincount/minlength?
(model/category_encoding/bincount/MaximumMaximum3model/category_encoding/bincount/minlength:output:0(model/category_encoding/bincount/mul:z:0*
T0	*
_output_shapes
: 2*
(model/category_encoding/bincount/Maximum?
*model/category_encoding/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2,
*model/category_encoding/bincount/maxlength?
(model/category_encoding/bincount/MinimumMinimum3model/category_encoding/bincount/maxlength:output:0,model/category_encoding/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2*
(model/category_encoding/bincount/Minimum?
(model/category_encoding/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2*
(model/category_encoding/bincount/Const_2?
.model/category_encoding/bincount/DenseBincountDenseBincountEmodel/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0,model/category_encoding/bincount/Minimum:z:01model/category_encoding/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(20
.model/category_encoding/bincount/DenseBincount?
model/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
model/concatenate_1/concat/axis?
model/concatenate_1/concatConcatV2model/normalization/truediv:z:07model/category_encoding/bincount/DenseBincount:output:0(model/concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
model/concatenate_1/concat?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#model/concatenate_1/concat:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd?
sequential/dense/SigmoidSigmoid!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense/Sigmoid?
sequential/dense/mulMul!sequential/dense/BiasAdd:output:0sequential/dense/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
sequential/dense/mul?
sequential/dense/IdentityIdentitysequential/dense/mul:z:0*
T0*(
_output_shapes
:??????????2
sequential/dense/Identity?
sequential/dense/IdentityN	IdentityNsequential/dense/mul:z:0!sequential/dense/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-48024*<
_output_shapes*
(:??????????:??????????2
sequential/dense/IdentityN?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul#sequential/dense/IdentityN:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential/dense_1/BiasAdd?
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02*
(sequential/dense_2/MatMul/ReadVariableOp?
sequential/dense_2/MatMulMatMul#sequential/dense_1/BiasAdd:output:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential/dense_2/MatMul?
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOp?
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential/dense_2/BiasAdd?
(sequential/dense_3/MatMul/ReadVariableOpReadVariableOp1sequential_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02*
(sequential/dense_3/MatMul/ReadVariableOp?
sequential/dense_3/MatMulMatMul#sequential/dense_2/BiasAdd:output:00sequential/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_3/MatMul?
)sequential/dense_3/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_3/BiasAdd/ReadVariableOp?
sequential/dense_3/BiasAddBiasAdd#sequential/dense_3/MatMul:product:01sequential/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_3/BiasAdd?
IdentityIdentity#sequential/dense_3/BiasAdd:output:0&^model/category_encoding/Assert/Assert+^model/normalization/Reshape/ReadVariableOp-^model/normalization/Reshape_1/ReadVariableOp=^model/string_lookup/None_lookup_table_find/LookupTableFindV2(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp*^sequential/dense_3/BiasAdd/ReadVariableOp)^sequential/dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2N
%model/category_encoding/Assert/Assert%model/category_encoding/Assert/Assert2X
*model/normalization/Reshape/ReadVariableOp*model/normalization/Reshape/ReadVariableOp2\
,model/normalization/Reshape_1/ReadVariableOp,model/normalization/Reshape_1/ReadVariableOp2|
<model/string_lookup/None_lookup_table_find/LookupTableFindV2<model/string_lookup/None_lookup_table_find/LookupTableFindV22R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2V
)sequential/dense_3/BiasAdd/ReadVariableOp)sequential/dense_3/BiasAdd/ReadVariableOp2T
(sequential/dense_3/MatMul/ReadVariableOp(sequential/dense_3/MatMul/ReadVariableOp:T P
'
_output_shapes
:?????????
%
_user_specified_nameinputs/area:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/floor:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/month_sold:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/rent:UQ
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
??
?
 __inference__wrapped_model_46990
area	
floor

month_sold
rent	
rooms
square_meter
year_of_construction
	year_soldU
Qmodel_1_model_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handleV
Rmodel_1_model_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value	I
;model_1_model_normalization_reshape_readvariableop_resource:K
=model_1_model_normalization_reshape_1_readvariableop_resource:K
7model_1_sequential_dense_matmul_readvariableop_resource:
??G
8model_1_sequential_dense_biasadd_readvariableop_resource:	?L
9model_1_sequential_dense_1_matmul_readvariableop_resource:	?@H
:model_1_sequential_dense_1_biasadd_readvariableop_resource:@K
9model_1_sequential_dense_2_matmul_readvariableop_resource:@ H
:model_1_sequential_dense_2_biasadd_readvariableop_resource: K
9model_1_sequential_dense_3_matmul_readvariableop_resource: H
:model_1_sequential_dense_3_biasadd_readvariableop_resource:
identity??-model_1/model/category_encoding/Assert/Assert?2model_1/model/normalization/Reshape/ReadVariableOp?4model_1/model/normalization/Reshape_1/ReadVariableOp?Dmodel_1/model/string_lookup/None_lookup_table_find/LookupTableFindV2?/model_1/sequential/dense/BiasAdd/ReadVariableOp?.model_1/sequential/dense/MatMul/ReadVariableOp?1model_1/sequential/dense_1/BiasAdd/ReadVariableOp?0model_1/sequential/dense_1/MatMul/ReadVariableOp?1model_1/sequential/dense_2/BiasAdd/ReadVariableOp?0model_1/sequential/dense_2/MatMul/ReadVariableOp?1model_1/sequential/dense_3/BiasAdd/ReadVariableOp?0model_1/sequential/dense_3/MatMul/ReadVariableOp?
Dmodel_1/model/string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Qmodel_1_model_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handleareaRmodel_1_model_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2F
Dmodel_1/model/string_lookup/None_lookup_table_find/LookupTableFindV2?
%model_1/model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%model_1/model/concatenate/concat/axis?
 model_1/model/concatenate/concatConcatV2floorroomssquare_meterrentyear_of_construction	year_sold
month_sold.model_1/model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2"
 model_1/model/concatenate/concat?
2model_1/model/normalization/Reshape/ReadVariableOpReadVariableOp;model_1_model_normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype024
2model_1/model/normalization/Reshape/ReadVariableOp?
)model_1/model/normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2+
)model_1/model/normalization/Reshape/shape?
#model_1/model/normalization/ReshapeReshape:model_1/model/normalization/Reshape/ReadVariableOp:value:02model_1/model/normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2%
#model_1/model/normalization/Reshape?
4model_1/model/normalization/Reshape_1/ReadVariableOpReadVariableOp=model_1_model_normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype026
4model_1/model/normalization/Reshape_1/ReadVariableOp?
+model_1/model/normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+model_1/model/normalization/Reshape_1/shape?
%model_1/model/normalization/Reshape_1Reshape<model_1/model/normalization/Reshape_1/ReadVariableOp:value:04model_1/model/normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2'
%model_1/model/normalization/Reshape_1?
model_1/model/normalization/subSub)model_1/model/concatenate/concat:output:0,model_1/model/normalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2!
model_1/model/normalization/sub?
 model_1/model/normalization/SqrtSqrt.model_1/model/normalization/Reshape_1:output:0*
T0*
_output_shapes

:2"
 model_1/model/normalization/Sqrt?
%model_1/model/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32'
%model_1/model/normalization/Maximum/y?
#model_1/model/normalization/MaximumMaximum$model_1/model/normalization/Sqrt:y:0.model_1/model/normalization/Maximum/y:output:0*
T0*
_output_shapes

:2%
#model_1/model/normalization/Maximum?
#model_1/model/normalization/truedivRealDiv#model_1/model/normalization/sub:z:0'model_1/model/normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????2%
#model_1/model/normalization/truediv?
%model_1/model/category_encoding/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%model_1/model/category_encoding/Const?
#model_1/model/category_encoding/MaxMaxMmodel_1/model/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0.model_1/model/category_encoding/Const:output:0*
T0	*
_output_shapes
: 2%
#model_1/model/category_encoding/Max?
'model_1/model/category_encoding/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'model_1/model/category_encoding/Const_1?
#model_1/model/category_encoding/MinMinMmodel_1/model/string_lookup/None_lookup_table_find/LookupTableFindV2:values:00model_1/model/category_encoding/Const_1:output:0*
T0	*
_output_shapes
: 2%
#model_1/model/category_encoding/Min?
&model_1/model/category_encoding/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2(
&model_1/model/category_encoding/Cast/x?
$model_1/model/category_encoding/CastCast/model_1/model/category_encoding/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2&
$model_1/model/category_encoding/Cast?
'model_1/model/category_encoding/GreaterGreater(model_1/model/category_encoding/Cast:y:0,model_1/model/category_encoding/Max:output:0*
T0	*
_output_shapes
: 2)
'model_1/model/category_encoding/Greater?
(model_1/model/category_encoding/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_1/model/category_encoding/Cast_1/x?
&model_1/model/category_encoding/Cast_1Cast1model_1/model/category_encoding/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2(
&model_1/model/category_encoding/Cast_1?
,model_1/model/category_encoding/GreaterEqualGreaterEqual,model_1/model/category_encoding/Min:output:0*model_1/model/category_encoding/Cast_1:y:0*
T0	*
_output_shapes
: 2.
,model_1/model/category_encoding/GreaterEqual?
*model_1/model/category_encoding/LogicalAnd
LogicalAnd+model_1/model/category_encoding/Greater:z:00model_1/model/category_encoding/GreaterEqual:z:0*
_output_shapes
: 2,
*model_1/model/category_encoding/LogicalAnd?
,model_1/model/category_encoding/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=1812.
,model_1/model/category_encoding/Assert/Const?
4model_1/model/category_encoding/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=18126
4model_1/model/category_encoding/Assert/Assert/data_0?
-model_1/model/category_encoding/Assert/AssertAssert.model_1/model/category_encoding/LogicalAnd:z:0=model_1/model/category_encoding/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2/
-model_1/model/category_encoding/Assert/Assert?
.model_1/model/category_encoding/bincount/ShapeShapeMmodel_1/model/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:20
.model_1/model/category_encoding/bincount/Shape?
.model_1/model/category_encoding/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 20
.model_1/model/category_encoding/bincount/Const?
-model_1/model/category_encoding/bincount/ProdProd7model_1/model/category_encoding/bincount/Shape:output:07model_1/model/category_encoding/bincount/Const:output:0*
T0*
_output_shapes
: 2/
-model_1/model/category_encoding/bincount/Prod?
2model_1/model/category_encoding/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 24
2model_1/model/category_encoding/bincount/Greater/y?
0model_1/model/category_encoding/bincount/GreaterGreater6model_1/model/category_encoding/bincount/Prod:output:0;model_1/model/category_encoding/bincount/Greater/y:output:0*
T0*
_output_shapes
: 22
0model_1/model/category_encoding/bincount/Greater?
-model_1/model/category_encoding/bincount/CastCast4model_1/model/category_encoding/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2/
-model_1/model/category_encoding/bincount/Cast?
0model_1/model/category_encoding/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0model_1/model/category_encoding/bincount/Const_1?
,model_1/model/category_encoding/bincount/MaxMaxMmodel_1/model/string_lookup/None_lookup_table_find/LookupTableFindV2:values:09model_1/model/category_encoding/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2.
,model_1/model/category_encoding/bincount/Max?
.model_1/model/category_encoding/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R20
.model_1/model/category_encoding/bincount/add/y?
,model_1/model/category_encoding/bincount/addAddV25model_1/model/category_encoding/bincount/Max:output:07model_1/model/category_encoding/bincount/add/y:output:0*
T0	*
_output_shapes
: 2.
,model_1/model/category_encoding/bincount/add?
,model_1/model/category_encoding/bincount/mulMul1model_1/model/category_encoding/bincount/Cast:y:00model_1/model/category_encoding/bincount/add:z:0*
T0	*
_output_shapes
: 2.
,model_1/model/category_encoding/bincount/mul?
2model_1/model/category_encoding/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?24
2model_1/model/category_encoding/bincount/minlength?
0model_1/model/category_encoding/bincount/MaximumMaximum;model_1/model/category_encoding/bincount/minlength:output:00model_1/model/category_encoding/bincount/mul:z:0*
T0	*
_output_shapes
: 22
0model_1/model/category_encoding/bincount/Maximum?
2model_1/model/category_encoding/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?24
2model_1/model/category_encoding/bincount/maxlength?
0model_1/model/category_encoding/bincount/MinimumMinimum;model_1/model/category_encoding/bincount/maxlength:output:04model_1/model/category_encoding/bincount/Maximum:z:0*
T0	*
_output_shapes
: 22
0model_1/model/category_encoding/bincount/Minimum?
0model_1/model/category_encoding/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 22
0model_1/model/category_encoding/bincount/Const_2?
6model_1/model/category_encoding/bincount/DenseBincountDenseBincountMmodel_1/model/string_lookup/None_lookup_table_find/LookupTableFindV2:values:04model_1/model/category_encoding/bincount/Minimum:z:09model_1/model/category_encoding/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(28
6model_1/model/category_encoding/bincount/DenseBincount?
'model_1/model/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2)
'model_1/model/concatenate_1/concat/axis?
"model_1/model/concatenate_1/concatConcatV2'model_1/model/normalization/truediv:z:0?model_1/model/category_encoding/bincount/DenseBincount:output:00model_1/model/concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2$
"model_1/model/concatenate_1/concat?
.model_1/sequential/dense/MatMul/ReadVariableOpReadVariableOp7model_1_sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.model_1/sequential/dense/MatMul/ReadVariableOp?
model_1/sequential/dense/MatMulMatMul+model_1/model/concatenate_1/concat:output:06model_1/sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
model_1/sequential/dense/MatMul?
/model_1/sequential/dense/BiasAdd/ReadVariableOpReadVariableOp8model_1_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/model_1/sequential/dense/BiasAdd/ReadVariableOp?
 model_1/sequential/dense/BiasAddBiasAdd)model_1/sequential/dense/MatMul:product:07model_1/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 model_1/sequential/dense/BiasAdd?
 model_1/sequential/dense/SigmoidSigmoid)model_1/sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2"
 model_1/sequential/dense/Sigmoid?
model_1/sequential/dense/mulMul)model_1/sequential/dense/BiasAdd:output:0$model_1/sequential/dense/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
model_1/sequential/dense/mul?
!model_1/sequential/dense/IdentityIdentity model_1/sequential/dense/mul:z:0*
T0*(
_output_shapes
:??????????2#
!model_1/sequential/dense/Identity?
"model_1/sequential/dense/IdentityN	IdentityN model_1/sequential/dense/mul:z:0)model_1/sequential/dense/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-46965*<
_output_shapes*
(:??????????:??????????2$
"model_1/sequential/dense/IdentityN?
0model_1/sequential/dense_1/MatMul/ReadVariableOpReadVariableOp9model_1_sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype022
0model_1/sequential/dense_1/MatMul/ReadVariableOp?
!model_1/sequential/dense_1/MatMulMatMul+model_1/sequential/dense/IdentityN:output:08model_1/sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2#
!model_1/sequential/dense_1/MatMul?
1model_1/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp:model_1_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1model_1/sequential/dense_1/BiasAdd/ReadVariableOp?
"model_1/sequential/dense_1/BiasAddBiasAdd+model_1/sequential/dense_1/MatMul:product:09model_1/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2$
"model_1/sequential/dense_1/BiasAdd?
0model_1/sequential/dense_2/MatMul/ReadVariableOpReadVariableOp9model_1_sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype022
0model_1/sequential/dense_2/MatMul/ReadVariableOp?
!model_1/sequential/dense_2/MatMulMatMul+model_1/sequential/dense_1/BiasAdd:output:08model_1/sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2#
!model_1/sequential/dense_2/MatMul?
1model_1/sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp:model_1_sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1model_1/sequential/dense_2/BiasAdd/ReadVariableOp?
"model_1/sequential/dense_2/BiasAddBiasAdd+model_1/sequential/dense_2/MatMul:product:09model_1/sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2$
"model_1/sequential/dense_2/BiasAdd?
0model_1/sequential/dense_3/MatMul/ReadVariableOpReadVariableOp9model_1_sequential_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype022
0model_1/sequential/dense_3/MatMul/ReadVariableOp?
!model_1/sequential/dense_3/MatMulMatMul+model_1/sequential/dense_2/BiasAdd:output:08model_1/sequential/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!model_1/sequential/dense_3/MatMul?
1model_1/sequential/dense_3/BiasAdd/ReadVariableOpReadVariableOp:model_1_sequential_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1model_1/sequential/dense_3/BiasAdd/ReadVariableOp?
"model_1/sequential/dense_3/BiasAddBiasAdd+model_1/sequential/dense_3/MatMul:product:09model_1/sequential/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2$
"model_1/sequential/dense_3/BiasAdd?
IdentityIdentity+model_1/sequential/dense_3/BiasAdd:output:0.^model_1/model/category_encoding/Assert/Assert3^model_1/model/normalization/Reshape/ReadVariableOp5^model_1/model/normalization/Reshape_1/ReadVariableOpE^model_1/model/string_lookup/None_lookup_table_find/LookupTableFindV20^model_1/sequential/dense/BiasAdd/ReadVariableOp/^model_1/sequential/dense/MatMul/ReadVariableOp2^model_1/sequential/dense_1/BiasAdd/ReadVariableOp1^model_1/sequential/dense_1/MatMul/ReadVariableOp2^model_1/sequential/dense_2/BiasAdd/ReadVariableOp1^model_1/sequential/dense_2/MatMul/ReadVariableOp2^model_1/sequential/dense_3/BiasAdd/ReadVariableOp1^model_1/sequential/dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2^
-model_1/model/category_encoding/Assert/Assert-model_1/model/category_encoding/Assert/Assert2h
2model_1/model/normalization/Reshape/ReadVariableOp2model_1/model/normalization/Reshape/ReadVariableOp2l
4model_1/model/normalization/Reshape_1/ReadVariableOp4model_1/model/normalization/Reshape_1/ReadVariableOp2?
Dmodel_1/model/string_lookup/None_lookup_table_find/LookupTableFindV2Dmodel_1/model/string_lookup/None_lookup_table_find/LookupTableFindV22b
/model_1/sequential/dense/BiasAdd/ReadVariableOp/model_1/sequential/dense/BiasAdd/ReadVariableOp2`
.model_1/sequential/dense/MatMul/ReadVariableOp.model_1/sequential/dense/MatMul/ReadVariableOp2f
1model_1/sequential/dense_1/BiasAdd/ReadVariableOp1model_1/sequential/dense_1/BiasAdd/ReadVariableOp2d
0model_1/sequential/dense_1/MatMul/ReadVariableOp0model_1/sequential/dense_1/MatMul/ReadVariableOp2f
1model_1/sequential/dense_2/BiasAdd/ReadVariableOp1model_1/sequential/dense_2/BiasAdd/ReadVariableOp2d
0model_1/sequential/dense_2/MatMul/ReadVariableOp0model_1/sequential/dense_2/MatMul/ReadVariableOp2f
1model_1/sequential/dense_3/BiasAdd/ReadVariableOp1model_1/sequential/dense_3/BiasAdd/ReadVariableOp2d
0model_1/sequential/dense_3/MatMul/ReadVariableOp0model_1/sequential/dense_3/MatMul/ReadVariableOp:M I
'
_output_shapes
:?????????

_user_specified_namearea:NJ
'
_output_shapes
:?????????

_user_specified_namefloor:SO
'
_output_shapes
:?????????
$
_user_specified_name
month_sold:MI
'
_output_shapes
:?????????

_user_specified_namerent:NJ
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
Y
-__inference_concatenate_1_layer_call_fn_48566
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
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_470782
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????:??????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?S
?
@__inference_model_layer_call_and_return_conditional_losses_48334
inputs_area
inputs_floor
inputs_month_sold
inputs_rent
inputs_rooms
inputs_square_meter
inputs_year_of_construction
inputs_year_soldG
Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handleH
Dstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value	;
-normalization_reshape_readvariableop_resource:=
/normalization_reshape_1_readvariableop_resource:
identity??category_encoding/Assert/Assert?$normalization/Reshape/ReadVariableOp?&normalization/Reshape_1/ReadVariableOp?6string_lookup/None_lookup_table_find/LookupTableFindV2?
6string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handleinputs_areaDstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????28
6string_lookup/None_lookup_table_find/LookupTableFindV2t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2inputs_floorinputs_roomsinputs_square_meterinputs_rentinputs_year_of_constructioninputs_year_soldinputs_month_sold concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate/concat?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape_1?
normalization/subSubconcatenate/concat:output:0normalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization/sub{
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization/truediv?
category_encoding/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding/Const?
category_encoding/MaxMax?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0 category_encoding/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding/Max?
category_encoding/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding/Const_1?
category_encoding/MinMin?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0"category_encoding/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding/Minw
category_encoding/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding/Cast/x?
category_encoding/CastCast!category_encoding/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding/Cast?
category_encoding/GreaterGreatercategory_encoding/Cast:y:0category_encoding/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding/Greaterz
category_encoding/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding/Cast_1/x?
category_encoding/Cast_1Cast#category_encoding/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding/Cast_1?
category_encoding/GreaterEqualGreaterEqualcategory_encoding/Min:output:0category_encoding/Cast_1:y:0*
T0	*
_output_shapes
: 2 
category_encoding/GreaterEqual?
category_encoding/LogicalAnd
LogicalAndcategory_encoding/Greater:z:0"category_encoding/GreaterEqual:z:0*
_output_shapes
: 2
category_encoding/LogicalAnd?
category_encoding/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=1812 
category_encoding/Assert/Const?
&category_encoding/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=1812(
&category_encoding/Assert/Assert/data_0?
category_encoding/Assert/AssertAssert category_encoding/LogicalAnd:z:0/category_encoding/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2!
category_encoding/Assert/Assert?
 category_encoding/bincount/ShapeShape?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2"
 category_encoding/bincount/Shape?
 category_encoding/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 category_encoding/bincount/Const?
category_encoding/bincount/ProdProd)category_encoding/bincount/Shape:output:0)category_encoding/bincount/Const:output:0*
T0*
_output_shapes
: 2!
category_encoding/bincount/Prod?
$category_encoding/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2&
$category_encoding/bincount/Greater/y?
"category_encoding/bincount/GreaterGreater(category_encoding/bincount/Prod:output:0-category_encoding/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2$
"category_encoding/bincount/Greater?
category_encoding/bincount/CastCast&category_encoding/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2!
category_encoding/bincount/Cast?
"category_encoding/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"category_encoding/bincount/Const_1?
category_encoding/bincount/MaxMax?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0+category_encoding/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/Max?
 category_encoding/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2"
 category_encoding/bincount/add/y?
category_encoding/bincount/addAddV2'category_encoding/bincount/Max:output:0)category_encoding/bincount/add/y:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/add?
category_encoding/bincount/mulMul#category_encoding/bincount/Cast:y:0"category_encoding/bincount/add:z:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/mul?
$category_encoding/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2&
$category_encoding/bincount/minlength?
"category_encoding/bincount/MaximumMaximum-category_encoding/bincount/minlength:output:0"category_encoding/bincount/mul:z:0*
T0	*
_output_shapes
: 2$
"category_encoding/bincount/Maximum?
$category_encoding/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2&
$category_encoding/bincount/maxlength?
"category_encoding/bincount/MinimumMinimum-category_encoding/bincount/maxlength:output:0&category_encoding/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2$
"category_encoding/bincount/Minimum?
"category_encoding/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2$
"category_encoding/bincount/Const_2?
(category_encoding/bincount/DenseBincountDenseBincount?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0&category_encoding/bincount/Minimum:z:0+category_encoding/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2*
(category_encoding/bincount/DenseBincountx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2normalization/truediv:z:01category_encoding/bincount/DenseBincount:output:0"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate_1/concat?
IdentityIdentityconcatenate_1/concat:output:0 ^category_encoding/Assert/Assert%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp7^string_lookup/None_lookup_table_find/LookupTableFindV2*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 2B
category_encoding/Assert/Assertcategory_encoding/Assert/Assert2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp2p
6string_lookup/None_lookup_table_find/LookupTableFindV26string_lookup/None_lookup_table_find/LookupTableFindV2:T P
'
_output_shapes
:?????????
%
_user_specified_nameinputs/area:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/floor:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/month_sold:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/rent:UQ
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
B__inference_dense_1_layer_call_and_return_conditional_losses_48601

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
__inference_restore_fn_48690
restored_tensors_0
restored_tensors_1	L
Hstring_lookup_index_table_table_restore_lookuptableimportv2_table_handle
identity??;string_lookup_index_table_table_restore/LookupTableImportV2?
;string_lookup_index_table_table_restore/LookupTableImportV2LookupTableImportV2Hstring_lookup_index_table_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 2=
;string_lookup_index_table_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Const?
IdentityIdentityConst:output:0<^string_lookup_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2z
;string_lookup_index_table_table_restore/LookupTableImportV2;string_lookup_index_table_table_restore/LookupTableImportV2:L H
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
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_47426

inputs
dense_47372:
??
dense_47374:	? 
dense_1_47388:	?@
dense_1_47390:@
dense_2_47404:@ 
dense_2_47406: 
dense_3_47420: 
dense_3_47422:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_47372dense_47374*
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
GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_473712
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_47388dense_1_47390*
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
GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_473872!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_47404dense_2_47406*
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
GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_474032!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_47420dense_3_47422*
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
GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_474192!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_47958
area	
floor

month_sold
rent	
rooms
square_meter
year_of_construction
	year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?@
	unknown_6:@
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallareafloor
month_soldrentroomssquare_meteryear_of_construction	year_soldunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_469902
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

_user_specified_namearea:NJ
'
_output_shapes
:?????????

_user_specified_namefloor:SO
'
_output_shapes
:?????????
$
_user_specified_name
month_sold:MI
'
_output_shapes
:?????????

_user_specified_namerent:NJ
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
*__inference_sequential_layer_call_fn_47572
dense_input
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_475322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:??????????
%
_user_specified_namedense_input
?	
?
B__inference_dense_2_layer_call_and_return_conditional_losses_48620

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
?
?
@__inference_dense_layer_call_and_return_conditional_losses_48582

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
_gradient_op_typeCustomGradient-48575*<
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_concatenate_layer_call_fn_48507
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
GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_470262
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
?
?
%__inference_model_layer_call_fn_47092
area	
floor

month_sold
rent	
rooms
square_meter
year_of_construction
	year_sold
unknown
	unknown_0	
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallareafloor
month_soldrentroomssquare_meteryear_of_construction	year_soldunknown	unknown_0	unknown_1	unknown_2*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_470812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
'
_output_shapes
:?????????

_user_specified_namearea:NJ
'
_output_shapes
:?????????

_user_specified_namefloor:SO
'
_output_shapes
:?????????
$
_user_specified_name
month_sold:MI
'
_output_shapes
:?????????

_user_specified_namerent:NJ
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
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_47596
dense_input
dense_47575:
??
dense_47577:	? 
dense_1_47580:	?@
dense_1_47582:@
dense_2_47585:@ 
dense_2_47587: 
dense_3_47590: 
dense_3_47592:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_47575dense_47577*
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
GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_473712
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_47580dense_1_47582*
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
GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_473872!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_47585dense_2_47587*
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
GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_474032!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_47590dense_3_47592*
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
GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_474192!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:U Q
(
_output_shapes
:??????????
%
_user_specified_namedense_input
?
.
__inference__initializer_48658
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
?
B__inference_model_1_layer_call_and_return_conditional_losses_48140
inputs_area
inputs_floor
inputs_month_sold
inputs_rent
inputs_rooms
inputs_square_meter
inputs_year_of_construction
inputs_year_soldM
Imodel_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handleN
Jmodel_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value	A
3model_normalization_reshape_readvariableop_resource:C
5model_normalization_reshape_1_readvariableop_resource:C
/sequential_dense_matmul_readvariableop_resource:
???
0sequential_dense_biasadd_readvariableop_resource:	?D
1sequential_dense_1_matmul_readvariableop_resource:	?@@
2sequential_dense_1_biasadd_readvariableop_resource:@C
1sequential_dense_2_matmul_readvariableop_resource:@ @
2sequential_dense_2_biasadd_readvariableop_resource: C
1sequential_dense_3_matmul_readvariableop_resource: @
2sequential_dense_3_biasadd_readvariableop_resource:
identity??%model/category_encoding/Assert/Assert?*model/normalization/Reshape/ReadVariableOp?,model/normalization/Reshape_1/ReadVariableOp?<model/string_lookup/None_lookup_table_find/LookupTableFindV2?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?)sequential/dense_2/BiasAdd/ReadVariableOp?(sequential/dense_2/MatMul/ReadVariableOp?)sequential/dense_3/BiasAdd/ReadVariableOp?(sequential/dense_3/MatMul/ReadVariableOp?
<model/string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Imodel_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handleinputs_areaJmodel_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2>
<model/string_lookup/None_lookup_table_find/LookupTableFindV2?
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis?
model/concatenate/concatConcatV2inputs_floorinputs_roomsinputs_square_meterinputs_rentinputs_year_of_constructioninputs_year_soldinputs_month_sold&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
model/concatenate/concat?
*model/normalization/Reshape/ReadVariableOpReadVariableOp3model_normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02,
*model/normalization/Reshape/ReadVariableOp?
!model/normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2#
!model/normalization/Reshape/shape?
model/normalization/ReshapeReshape2model/normalization/Reshape/ReadVariableOp:value:0*model/normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
model/normalization/Reshape?
,model/normalization/Reshape_1/ReadVariableOpReadVariableOp5model_normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,model/normalization/Reshape_1/ReadVariableOp?
#model/normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2%
#model/normalization/Reshape_1/shape?
model/normalization/Reshape_1Reshape4model/normalization/Reshape_1/ReadVariableOp:value:0,model/normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
model/normalization/Reshape_1?
model/normalization/subSub!model/concatenate/concat:output:0$model/normalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2
model/normalization/sub?
model/normalization/SqrtSqrt&model/normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
model/normalization/Sqrt?
model/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
model/normalization/Maximum/y?
model/normalization/MaximumMaximummodel/normalization/Sqrt:y:0&model/normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
model/normalization/Maximum?
model/normalization/truedivRealDivmodel/normalization/sub:z:0model/normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????2
model/normalization/truediv?
model/category_encoding/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
model/category_encoding/Const?
model/category_encoding/MaxMaxEmodel/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0&model/category_encoding/Const:output:0*
T0	*
_output_shapes
: 2
model/category_encoding/Max?
model/category_encoding/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
model/category_encoding/Const_1?
model/category_encoding/MinMinEmodel/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0(model/category_encoding/Const_1:output:0*
T0	*
_output_shapes
: 2
model/category_encoding/Min?
model/category_encoding/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2 
model/category_encoding/Cast/x?
model/category_encoding/CastCast'model/category_encoding/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
model/category_encoding/Cast?
model/category_encoding/GreaterGreater model/category_encoding/Cast:y:0$model/category_encoding/Max:output:0*
T0	*
_output_shapes
: 2!
model/category_encoding/Greater?
 model/category_encoding/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2"
 model/category_encoding/Cast_1/x?
model/category_encoding/Cast_1Cast)model/category_encoding/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2 
model/category_encoding/Cast_1?
$model/category_encoding/GreaterEqualGreaterEqual$model/category_encoding/Min:output:0"model/category_encoding/Cast_1:y:0*
T0	*
_output_shapes
: 2&
$model/category_encoding/GreaterEqual?
"model/category_encoding/LogicalAnd
LogicalAnd#model/category_encoding/Greater:z:0(model/category_encoding/GreaterEqual:z:0*
_output_shapes
: 2$
"model/category_encoding/LogicalAnd?
$model/category_encoding/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=1812&
$model/category_encoding/Assert/Const?
,model/category_encoding/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=1812.
,model/category_encoding/Assert/Assert/data_0?
%model/category_encoding/Assert/AssertAssert&model/category_encoding/LogicalAnd:z:05model/category_encoding/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2'
%model/category_encoding/Assert/Assert?
&model/category_encoding/bincount/ShapeShapeEmodel/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2(
&model/category_encoding/bincount/Shape?
&model/category_encoding/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model/category_encoding/bincount/Const?
%model/category_encoding/bincount/ProdProd/model/category_encoding/bincount/Shape:output:0/model/category_encoding/bincount/Const:output:0*
T0*
_output_shapes
: 2'
%model/category_encoding/bincount/Prod?
*model/category_encoding/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2,
*model/category_encoding/bincount/Greater/y?
(model/category_encoding/bincount/GreaterGreater.model/category_encoding/bincount/Prod:output:03model/category_encoding/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2*
(model/category_encoding/bincount/Greater?
%model/category_encoding/bincount/CastCast,model/category_encoding/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2'
%model/category_encoding/bincount/Cast?
(model/category_encoding/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2*
(model/category_encoding/bincount/Const_1?
$model/category_encoding/bincount/MaxMaxEmodel/string_lookup/None_lookup_table_find/LookupTableFindV2:values:01model/category_encoding/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2&
$model/category_encoding/bincount/Max?
&model/category_encoding/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&model/category_encoding/bincount/add/y?
$model/category_encoding/bincount/addAddV2-model/category_encoding/bincount/Max:output:0/model/category_encoding/bincount/add/y:output:0*
T0	*
_output_shapes
: 2&
$model/category_encoding/bincount/add?
$model/category_encoding/bincount/mulMul)model/category_encoding/bincount/Cast:y:0(model/category_encoding/bincount/add:z:0*
T0	*
_output_shapes
: 2&
$model/category_encoding/bincount/mul?
*model/category_encoding/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2,
*model/category_encoding/bincount/minlength?
(model/category_encoding/bincount/MaximumMaximum3model/category_encoding/bincount/minlength:output:0(model/category_encoding/bincount/mul:z:0*
T0	*
_output_shapes
: 2*
(model/category_encoding/bincount/Maximum?
*model/category_encoding/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2,
*model/category_encoding/bincount/maxlength?
(model/category_encoding/bincount/MinimumMinimum3model/category_encoding/bincount/maxlength:output:0,model/category_encoding/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2*
(model/category_encoding/bincount/Minimum?
(model/category_encoding/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2*
(model/category_encoding/bincount/Const_2?
.model/category_encoding/bincount/DenseBincountDenseBincountEmodel/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0,model/category_encoding/bincount/Minimum:z:01model/category_encoding/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(20
.model/category_encoding/bincount/DenseBincount?
model/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
model/concatenate_1/concat/axis?
model/concatenate_1/concatConcatV2model/normalization/truediv:z:07model/category_encoding/bincount/DenseBincount:output:0(model/concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
model/concatenate_1/concat?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#model/concatenate_1/concat:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd?
sequential/dense/SigmoidSigmoid!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense/Sigmoid?
sequential/dense/mulMul!sequential/dense/BiasAdd:output:0sequential/dense/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
sequential/dense/mul?
sequential/dense/IdentityIdentitysequential/dense/mul:z:0*
T0*(
_output_shapes
:??????????2
sequential/dense/Identity?
sequential/dense/IdentityN	IdentityNsequential/dense/mul:z:0!sequential/dense/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-48115*<
_output_shapes*
(:??????????:??????????2
sequential/dense/IdentityN?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul#sequential/dense/IdentityN:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential/dense_1/BiasAdd?
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02*
(sequential/dense_2/MatMul/ReadVariableOp?
sequential/dense_2/MatMulMatMul#sequential/dense_1/BiasAdd:output:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential/dense_2/MatMul?
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOp?
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential/dense_2/BiasAdd?
(sequential/dense_3/MatMul/ReadVariableOpReadVariableOp1sequential_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02*
(sequential/dense_3/MatMul/ReadVariableOp?
sequential/dense_3/MatMulMatMul#sequential/dense_2/BiasAdd:output:00sequential/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_3/MatMul?
)sequential/dense_3/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_3/BiasAdd/ReadVariableOp?
sequential/dense_3/BiasAddBiasAdd#sequential/dense_3/MatMul:product:01sequential/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_3/BiasAdd?
IdentityIdentity#sequential/dense_3/BiasAdd:output:0&^model/category_encoding/Assert/Assert+^model/normalization/Reshape/ReadVariableOp-^model/normalization/Reshape_1/ReadVariableOp=^model/string_lookup/None_lookup_table_find/LookupTableFindV2(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp*^sequential/dense_3/BiasAdd/ReadVariableOp)^sequential/dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2N
%model/category_encoding/Assert/Assert%model/category_encoding/Assert/Assert2X
*model/normalization/Reshape/ReadVariableOp*model/normalization/Reshape/ReadVariableOp2\
,model/normalization/Reshape_1/ReadVariableOp,model/normalization/Reshape_1/ReadVariableOp2|
<model/string_lookup/None_lookup_table_find/LookupTableFindV2<model/string_lookup/None_lookup_table_find/LookupTableFindV22R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2V
)sequential/dense_3/BiasAdd/ReadVariableOp)sequential/dense_3/BiasAdd/ReadVariableOp2T
(sequential/dense_3/MatMul/ReadVariableOp(sequential/dense_3/MatMul/ReadVariableOp:T P
'
_output_shapes
:?????????
%
_user_specified_nameinputs/area:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/floor:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/month_sold:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/rent:UQ
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
?,
?
__inference_adapt_step_48553
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
?
,
__inference__destroyer_48663
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
?S
?
@__inference_model_layer_call_and_return_conditional_losses_47289
area	
floor

month_sold
rent	
rooms
square_meter
year_of_construction
	year_soldG
Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handleH
Dstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value	;
-normalization_reshape_readvariableop_resource:=
/normalization_reshape_1_readvariableop_resource:
identity??category_encoding/Assert/Assert?$normalization/Reshape/ReadVariableOp?&normalization/Reshape_1/ReadVariableOp?6string_lookup/None_lookup_table_find/LookupTableFindV2?
6string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handleareaDstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????28
6string_lookup/None_lookup_table_find/LookupTableFindV2?
concatenate/PartitionedCallPartitionedCallfloorroomssquare_meterrentyear_of_construction	year_sold
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
GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_470262
concatenate/PartitionedCall?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape_1?
normalization/subSub$concatenate/PartitionedCall:output:0normalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization/sub{
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization/truediv?
category_encoding/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding/Const?
category_encoding/MaxMax?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0 category_encoding/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding/Max?
category_encoding/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding/Const_1?
category_encoding/MinMin?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0"category_encoding/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding/Minw
category_encoding/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding/Cast/x?
category_encoding/CastCast!category_encoding/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding/Cast?
category_encoding/GreaterGreatercategory_encoding/Cast:y:0category_encoding/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding/Greaterz
category_encoding/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding/Cast_1/x?
category_encoding/Cast_1Cast#category_encoding/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding/Cast_1?
category_encoding/GreaterEqualGreaterEqualcategory_encoding/Min:output:0category_encoding/Cast_1:y:0*
T0	*
_output_shapes
: 2 
category_encoding/GreaterEqual?
category_encoding/LogicalAnd
LogicalAndcategory_encoding/Greater:z:0"category_encoding/GreaterEqual:z:0*
_output_shapes
: 2
category_encoding/LogicalAnd?
category_encoding/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=1812 
category_encoding/Assert/Const?
&category_encoding/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=1812(
&category_encoding/Assert/Assert/data_0?
category_encoding/Assert/AssertAssert category_encoding/LogicalAnd:z:0/category_encoding/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2!
category_encoding/Assert/Assert?
 category_encoding/bincount/ShapeShape?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2"
 category_encoding/bincount/Shape?
 category_encoding/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 category_encoding/bincount/Const?
category_encoding/bincount/ProdProd)category_encoding/bincount/Shape:output:0)category_encoding/bincount/Const:output:0*
T0*
_output_shapes
: 2!
category_encoding/bincount/Prod?
$category_encoding/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2&
$category_encoding/bincount/Greater/y?
"category_encoding/bincount/GreaterGreater(category_encoding/bincount/Prod:output:0-category_encoding/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2$
"category_encoding/bincount/Greater?
category_encoding/bincount/CastCast&category_encoding/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2!
category_encoding/bincount/Cast?
"category_encoding/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"category_encoding/bincount/Const_1?
category_encoding/bincount/MaxMax?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0+category_encoding/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/Max?
 category_encoding/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2"
 category_encoding/bincount/add/y?
category_encoding/bincount/addAddV2'category_encoding/bincount/Max:output:0)category_encoding/bincount/add/y:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/add?
category_encoding/bincount/mulMul#category_encoding/bincount/Cast:y:0"category_encoding/bincount/add:z:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/mul?
$category_encoding/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2&
$category_encoding/bincount/minlength?
"category_encoding/bincount/MaximumMaximum-category_encoding/bincount/minlength:output:0"category_encoding/bincount/mul:z:0*
T0	*
_output_shapes
: 2$
"category_encoding/bincount/Maximum?
$category_encoding/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2&
$category_encoding/bincount/maxlength?
"category_encoding/bincount/MinimumMinimum-category_encoding/bincount/maxlength:output:0&category_encoding/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2$
"category_encoding/bincount/Minimum?
"category_encoding/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2$
"category_encoding/bincount/Const_2?
(category_encoding/bincount/DenseBincountDenseBincount?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0&category_encoding/bincount/Minimum:z:0+category_encoding/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2*
(category_encoding/bincount/DenseBincount?
concatenate_1/PartitionedCallPartitionedCallnormalization/truediv:z:01category_encoding/bincount/DenseBincount:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_470782
concatenate_1/PartitionedCall?
IdentityIdentity&concatenate_1/PartitionedCall:output:0 ^category_encoding/Assert/Assert%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp7^string_lookup/None_lookup_table_find/LookupTableFindV2*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 2B
category_encoding/Assert/Assertcategory_encoding/Assert/Assert2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp2p
6string_lookup/None_lookup_table_find/LookupTableFindV26string_lookup/None_lookup_table_find/LookupTableFindV2:M I
'
_output_shapes
:?????????

_user_specified_namearea:NJ
'
_output_shapes
:?????????

_user_specified_namefloor:SO
'
_output_shapes
:?????????
$
_user_specified_name
month_sold:MI
'
_output_shapes
:?????????

_user_specified_namerent:NJ
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
?
?
B__inference_model_1_layer_call_and_return_conditional_losses_47914
area	
floor

month_sold
rent	
rooms
square_meter
year_of_construction
	year_sold
model_47887
model_47889	
model_47891:
model_47893:$
sequential_47896:
??
sequential_47898:	?#
sequential_47900:	?@
sequential_47902:@"
sequential_47904:@ 
sequential_47906: "
sequential_47908: 
sequential_47910:
identity??model/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCallareafloor
month_soldrentroomssquare_meteryear_of_construction	year_soldmodel_47887model_47889model_47891model_47893*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_471992
model/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0sequential_47896sequential_47898sequential_47900sequential_47902sequential_47904sequential_47906sequential_47908sequential_47910*
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
GPU2*0J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_475322$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^model/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : 2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:M I
'
_output_shapes
:?????????

_user_specified_namearea:NJ
'
_output_shapes
:?????????

_user_specified_namefloor:SO
'
_output_shapes
:?????????
$
_user_specified_name
month_sold:MI
'
_output_shapes
:?????????

_user_specified_namerent:NJ
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
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_47532

inputs
dense_47511:
??
dense_47513:	? 
dense_1_47516:	?@
dense_1_47518:@
dense_2_47521:@ 
dense_2_47523: 
dense_3_47526: 
dense_3_47528:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_47511dense_47513*
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
GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_473712
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_47516dense_1_47518*
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
GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_473872!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_47521dense_2_47523*
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
GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_474032!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_47526dense_3_47528*
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
GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_474192!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
F__inference_concatenate_layer_call_and_return_conditional_losses_48496
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
?S
?
@__inference_model_layer_call_and_return_conditional_losses_48273
inputs_area
inputs_floor
inputs_month_sold
inputs_rent
inputs_rooms
inputs_square_meter
inputs_year_of_construction
inputs_year_soldG
Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handleH
Dstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value	;
-normalization_reshape_readvariableop_resource:=
/normalization_reshape_1_readvariableop_resource:
identity??category_encoding/Assert/Assert?$normalization/Reshape/ReadVariableOp?&normalization/Reshape_1/ReadVariableOp?6string_lookup/None_lookup_table_find/LookupTableFindV2?
6string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handleinputs_areaDstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????28
6string_lookup/None_lookup_table_find/LookupTableFindV2t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2inputs_floorinputs_roomsinputs_square_meterinputs_rentinputs_year_of_constructioninputs_year_soldinputs_month_sold concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate/concat?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape_1?
normalization/subSubconcatenate/concat:output:0normalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization/sub{
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????2
normalization/truediv?
category_encoding/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding/Const?
category_encoding/MaxMax?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0 category_encoding/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding/Max?
category_encoding/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding/Const_1?
category_encoding/MinMin?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0"category_encoding/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding/Minw
category_encoding/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding/Cast/x?
category_encoding/CastCast!category_encoding/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding/Cast?
category_encoding/GreaterGreatercategory_encoding/Cast:y:0category_encoding/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding/Greaterz
category_encoding/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding/Cast_1/x?
category_encoding/Cast_1Cast#category_encoding/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding/Cast_1?
category_encoding/GreaterEqualGreaterEqualcategory_encoding/Min:output:0category_encoding/Cast_1:y:0*
T0	*
_output_shapes
: 2 
category_encoding/GreaterEqual?
category_encoding/LogicalAnd
LogicalAndcategory_encoding/Greater:z:0"category_encoding/GreaterEqual:z:0*
_output_shapes
: 2
category_encoding/LogicalAnd?
category_encoding/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=1812 
category_encoding/Assert/Const?
&category_encoding/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=1812(
&category_encoding/Assert/Assert/data_0?
category_encoding/Assert/AssertAssert category_encoding/LogicalAnd:z:0/category_encoding/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2!
category_encoding/Assert/Assert?
 category_encoding/bincount/ShapeShape?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2"
 category_encoding/bincount/Shape?
 category_encoding/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 category_encoding/bincount/Const?
category_encoding/bincount/ProdProd)category_encoding/bincount/Shape:output:0)category_encoding/bincount/Const:output:0*
T0*
_output_shapes
: 2!
category_encoding/bincount/Prod?
$category_encoding/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2&
$category_encoding/bincount/Greater/y?
"category_encoding/bincount/GreaterGreater(category_encoding/bincount/Prod:output:0-category_encoding/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2$
"category_encoding/bincount/Greater?
category_encoding/bincount/CastCast&category_encoding/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2!
category_encoding/bincount/Cast?
"category_encoding/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"category_encoding/bincount/Const_1?
category_encoding/bincount/MaxMax?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0+category_encoding/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/Max?
 category_encoding/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2"
 category_encoding/bincount/add/y?
category_encoding/bincount/addAddV2'category_encoding/bincount/Max:output:0)category_encoding/bincount/add/y:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/add?
category_encoding/bincount/mulMul#category_encoding/bincount/Cast:y:0"category_encoding/bincount/add:z:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/mul?
$category_encoding/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2&
$category_encoding/bincount/minlength?
"category_encoding/bincount/MaximumMaximum-category_encoding/bincount/minlength:output:0"category_encoding/bincount/mul:z:0*
T0	*
_output_shapes
: 2$
"category_encoding/bincount/Maximum?
$category_encoding/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?2&
$category_encoding/bincount/maxlength?
"category_encoding/bincount/MinimumMinimum-category_encoding/bincount/maxlength:output:0&category_encoding/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2$
"category_encoding/bincount/Minimum?
"category_encoding/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2$
"category_encoding/bincount/Const_2?
(category_encoding/bincount/DenseBincountDenseBincount?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0&category_encoding/bincount/Minimum:z:0+category_encoding/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2*
(category_encoding/bincount/DenseBincountx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2normalization/truediv:z:01category_encoding/bincount/DenseBincount:output:0"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate_1/concat?
IdentityIdentityconcatenate_1/concat:output:0 ^category_encoding/Assert/Assert%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp7^string_lookup/None_lookup_table_find/LookupTableFindV2*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : 2B
category_encoding/Assert/Assertcategory_encoding/Assert/Assert2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp2p
6string_lookup/None_lookup_table_find/LookupTableFindV26string_lookup/None_lookup_table_find/LookupTableFindV2:T P
'
_output_shapes
:?????????
%
_user_specified_nameinputs/area:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/floor:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/month_sold:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/rent:UQ
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
7
floor.
serving_default_floor:0?????????
A

month_sold3
serving_default_month_sold:0?????????
5
rent-
serving_default_rent:0?????????
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
serving_default_year_sold:0?????????>

sequential0
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
_tf_keras_network??{"name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}, "name": "area", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "floor"}, "name": "floor", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}, "name": "month_sold", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rent"}, "name": "rent", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}, "name": "rooms", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}, "name": "square_meter", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_of_construction"}, "name": "year_of_construction", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}, "name": "year_sold", "inbound_nodes": [], "shared_object_id": 7}, {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "floor"}, "name": "floor", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}, "name": "rooms", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}, "name": "square_meter", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rent"}, "name": "rent", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_of_construction"}, "name": "year_of_construction", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}, "name": "year_sold", "inbound_nodes": [], "shared_object_id": 7}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}, "name": "month_sold", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}, "name": "area", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["floor", 0, 0, {}], ["rooms", 0, 0, {}], ["square_meter", 0, 0, {}], ["rent", 0, 0, {}], ["year_of_construction", 0, 0, {}], ["year_sold", 0, 0, {}], ["month_sold", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "output_mode": "int", "pad_to_max_tokens": false, "vocabulary_size": 181, "vocabulary": null, "encoding": "utf-8"}, "name": "string_lookup", "inbound_nodes": [[["area", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "name": "normalization", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding", "trainable": true, "dtype": "float32", "num_tokens": 181, "output_mode": "binary", "sparse": false}, "name": "category_encoding", "inbound_nodes": [[["string_lookup", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["normalization", 0, 0, {}], ["category_encoding", 0, 0, {}]]]}], "input_layers": {"area": ["area", 0, 0], "floor": ["floor", 0, 0], "rooms": ["rooms", 0, 0], "square_meter": ["square_meter", 0, 0], "rent": ["rent", 0, 0], "year_of_construction": ["year_of_construction", 0, 0], "year_sold": ["year_sold", 0, 0], "month_sold": ["month_sold", 0, 0]}, "output_layers": [["concatenate_1", 0, 0]]}, "name": "model", "inbound_nodes": [{"area": ["area", 0, 0, {}], "floor": ["floor", 0, 0, {}], "rooms": ["rooms", 0, 0, {}], "square_meter": ["square_meter", 0, 0, {}], "rent": ["rent", 0, 0, {}], "year_of_construction": ["year_of_construction", 0, 0, {}], "year_sold": ["year_sold", 0, 0, {}], "month_sold": ["month_sold", 0, 0, {}]}]}, {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 188]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential", "inbound_nodes": [[["model", 1, 0, {}]]]}], "input_layers": {"area": ["area", 0, 0], "floor": ["floor", 0, 0], "rooms": ["rooms", 0, 0], "square_meter": ["square_meter", 0, 0], "rent": ["rent", 0, 0], "year_of_construction": ["year_of_construction", 0, 0], "year_sold": ["year_sold", 0, 0], "month_sold": ["month_sold", 0, 0]}, "output_layers": [["sequential", 1, 0]]}, "shared_object_id": 28, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"area": {"class_name": "TensorShape", "items": [null, 1]}, "floor": {"class_name": "TensorShape", "items": [null, 1]}, "rooms": {"class_name": "TensorShape", "items": [null, 1]}, "square_meter": {"class_name": "TensorShape", "items": [null, 1]}, "rent": {"class_name": "TensorShape", "items": [null, 1]}, "year_of_construction": {"class_name": "TensorShape", "items": [null, 1]}, "year_sold": {"class_name": "TensorShape", "items": [null, 1]}, "month_sold": {"class_name": "TensorShape", "items": [null, 1]}}, "is_graph_network": true, "save_spec": {"area": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "string", "area"]}, "floor": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "floor"]}, "rooms": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "rooms"]}, "square_meter": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "square_meter"]}, "rent": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "rent"]}, "year_of_construction": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "year_of_construction"]}, "year_sold": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "year_sold"]}, "month_sold": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "month_sold"]}}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}, "name": "area", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "floor"}, "name": "floor", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}, "name": "month_sold", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rent"}, "name": "rent", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}, "name": "rooms", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}, "name": "square_meter", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_of_construction"}, "name": "year_of_construction", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}, "name": "year_sold", "inbound_nodes": [], "shared_object_id": 7}, {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "floor"}, "name": "floor", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}, "name": "rooms", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}, "name": "square_meter", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rent"}, "name": "rent", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_of_construction"}, "name": "year_of_construction", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}, "name": "year_sold", "inbound_nodes": [], "shared_object_id": 7}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}, "name": "month_sold", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}, "name": "area", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["floor", 0, 0, {}], ["rooms", 0, 0, {}], ["square_meter", 0, 0, {}], ["rent", 0, 0, {}], ["year_of_construction", 0, 0, {}], ["year_sold", 0, 0, {}], ["month_sold", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "output_mode": "int", "pad_to_max_tokens": false, "vocabulary_size": 181, "vocabulary": null, "encoding": "utf-8"}, "name": "string_lookup", "inbound_nodes": [[["area", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "name": "normalization", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding", "trainable": true, "dtype": "float32", "num_tokens": 181, "output_mode": "binary", "sparse": false}, "name": "category_encoding", "inbound_nodes": [[["string_lookup", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["normalization", 0, 0, {}], ["category_encoding", 0, 0, {}]]]}], "input_layers": {"area": ["area", 0, 0], "floor": ["floor", 0, 0], "rooms": ["rooms", 0, 0], "square_meter": ["square_meter", 0, 0], "rent": ["rent", 0, 0], "year_of_construction": ["year_of_construction", 0, 0], "year_sold": ["year_sold", 0, 0], "month_sold": ["month_sold", 0, 0]}, "output_layers": [["concatenate_1", 0, 0]]}, "name": "model", "inbound_nodes": [{"area": ["area", 0, 0, {}], "floor": ["floor", 0, 0, {}], "rooms": ["rooms", 0, 0, {}], "square_meter": ["square_meter", 0, 0, {}], "rent": ["rent", 0, 0, {}], "year_of_construction": ["year_of_construction", 0, 0, {}], "year_sold": ["year_sold", 0, 0, {}], "month_sold": ["month_sold", 0, 0, {}]}], "shared_object_id": 13}, {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 188]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential", "inbound_nodes": [[["model", 1, 0, {}]]], "shared_object_id": 27}], "input_layers": {"area": ["area", 0, 0], "floor": ["floor", 0, 0], "rooms": ["rooms", 0, 0], "square_meter": ["square_meter", 0, 0], "rent": ["rent", 0, 0], "year_of_construction": ["year_of_construction", 0, 0], "year_sold": ["year_sold", 0, 0], "month_sold": ["month_sold", 0, 0]}, "output_layers": [["sequential", 1, 0]]}}, "training_config": {"loss": {"class_name": "MeanAbsoluteError", "config": {"reduction": "auto", "name": "mean_absolute_error"}, "shared_object_id": 37}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "area", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "floor", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "floor"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "month_sold", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "rent", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rent"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "rooms", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "square_meter", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "year_of_construction", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_of_construction"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "year_sold", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}}
?_
layer-0
layer-1
layer-2
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
?__call__"?\
_tf_keras_network?\{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "floor"}, "name": "floor", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}, "name": "rooms", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}, "name": "square_meter", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rent"}, "name": "rent", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_of_construction"}, "name": "year_of_construction", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}, "name": "year_sold", "inbound_nodes": [], "shared_object_id": 7}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}, "name": "month_sold", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}, "name": "area", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["floor", 0, 0, {}], ["rooms", 0, 0, {}], ["square_meter", 0, 0, {}], ["rent", 0, 0, {}], ["year_of_construction", 0, 0, {}], ["year_sold", 0, 0, {}], ["month_sold", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "output_mode": "int", "pad_to_max_tokens": false, "vocabulary_size": 181, "vocabulary": null, "encoding": "utf-8"}, "name": "string_lookup", "inbound_nodes": [[["area", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "name": "normalization", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding", "trainable": true, "dtype": "float32", "num_tokens": 181, "output_mode": "binary", "sparse": false}, "name": "category_encoding", "inbound_nodes": [[["string_lookup", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["normalization", 0, 0, {}], ["category_encoding", 0, 0, {}]]]}], "input_layers": {"area": ["area", 0, 0], "floor": ["floor", 0, 0], "rooms": ["rooms", 0, 0], "square_meter": ["square_meter", 0, 0], "rent": ["rent", 0, 0], "year_of_construction": ["year_of_construction", 0, 0], "year_sold": ["year_sold", 0, 0], "month_sold": ["month_sold", 0, 0]}, "output_layers": [["concatenate_1", 0, 0]]}, "inbound_nodes": [{"area": ["area", 0, 0, {}], "floor": ["floor", 0, 0, {}], "rooms": ["rooms", 0, 0, {}], "square_meter": ["square_meter", 0, 0, {}], "rent": ["rent", 0, 0, {}], "year_of_construction": ["year_of_construction", 0, 0, {}], "year_sold": ["year_sold", 0, 0, {}], "month_sold": ["month_sold", 0, 0, {}]}], "shared_object_id": 13, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"area": {"class_name": "TensorShape", "items": [null, 1]}, "floor": {"class_name": "TensorShape", "items": [null, 1]}, "rooms": {"class_name": "TensorShape", "items": [null, 1]}, "square_meter": {"class_name": "TensorShape", "items": [null, 1]}, "rent": {"class_name": "TensorShape", "items": [null, 1]}, "year_of_construction": {"class_name": "TensorShape", "items": [null, 1]}, "year_sold": {"class_name": "TensorShape", "items": [null, 1]}, "month_sold": {"class_name": "TensorShape", "items": [null, 1]}}, "is_graph_network": true, "save_spec": {"area": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "string", "area"]}, "floor": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "floor"]}, "rooms": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "rooms"]}, "square_meter": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "square_meter"]}, "rent": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "rent"]}, "year_of_construction": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "year_of_construction"]}, "year_sold": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "year_sold"]}, "month_sold": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "month_sold"]}}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "floor"}, "name": "floor", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rooms"}, "name": "rooms", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "square_meter"}, "name": "square_meter", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rent"}, "name": "rent", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_of_construction"}, "name": "year_of_construction", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "year_sold"}, "name": "year_sold", "inbound_nodes": [], "shared_object_id": 7}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "month_sold"}, "name": "month_sold", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "area"}, "name": "area", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["floor", 0, 0, {}], ["rooms", 0, 0, {}], ["square_meter", 0, 0, {}], ["rent", 0, 0, {}], ["year_of_construction", 0, 0, {}], ["year_sold", 0, 0, {}], ["month_sold", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "StringLookup", "config": {"name": "string_lookup", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "output_mode": "int", "pad_to_max_tokens": false, "vocabulary_size": 181, "vocabulary": null, "encoding": "utf-8"}, "name": "string_lookup", "inbound_nodes": [[["area", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Normalization", "config": {"name": "normalization", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "name": "normalization", "inbound_nodes": [[["concatenate", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding", "trainable": true, "dtype": "float32", "num_tokens": 181, "output_mode": "binary", "sparse": false}, "name": "category_encoding", "inbound_nodes": [[["string_lookup", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["normalization", 0, 0, {}], ["category_encoding", 0, 0, {}]]], "shared_object_id": 12}], "input_layers": {"area": ["area", 0, 0], "floor": ["floor", 0, 0], "rooms": ["rooms", 0, 0], "square_meter": ["square_meter", 0, 0], "rent": ["rent", 0, 0], "year_of_construction": ["year_of_construction", 0, 0], "year_sold": ["year_sold", 0, 0], "month_sold": ["month_sold", 0, 0]}, "output_layers": [["concatenate_1", 0, 0]]}}}
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
_tf_keras_sequential?&{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 188]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "inbound_nodes": [[["model", 1, 0, {}]]], "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 188}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 188]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 188]}, "float32", "dense_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 188]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}, "shared_object_id": 14}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 23}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 26}]}}}
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
_tf_keras_layer?{"name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["floor", 0, 0, {}], ["rooms", 0, 0, {}], ["square_meter", 0, 0, {}], ["rent", 0, 0, {}], ["year_of_construction", 0, 0, {}], ["year_sold", 0, 0, {}], ["month_sold", 0, 0, {}]]], "shared_object_id": 8, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}]}
?
;state_variables

<_table
=	keras_api"?
_tf_keras_layer?{"name": "string_lookup", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": null, "stateful": true, "must_restore_from_config": true, "class_name": "StringLookup", "config": {"name": "string_lookup", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "output_mode": "int", "pad_to_max_tokens": false, "vocabulary_size": 181, "vocabulary": null, "encoding": "utf-8"}, "inbound_nodes": [[["area", 0, 0, {}]]], "shared_object_id": 9}
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
_tf_keras_layer?{"name": "normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": true, "must_restore_from_config": true, "class_name": "Normalization", "config": {"name": "normalization", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "inbound_nodes": [[["concatenate", 0, 0, {}]]], "shared_object_id": 10, "build_input_shape": [null, 7]}
?
C	keras_api"?
_tf_keras_layer?{"name": "category_encoding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "CategoryEncoding", "config": {"name": "category_encoding", "trainable": true, "dtype": "float32", "num_tokens": 181, "output_mode": "binary", "sparse": false}, "inbound_nodes": [[["string_lookup", 0, 0, {}]]], "shared_object_id": 11}
?
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "concatenate_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["normalization", 0, 0, {}], ["category_encoding", 0, 0, {}]]], "shared_object_id": 12, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 7]}, {"class_name": "TensorShape", "items": [null, 181]}]}
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
_tf_keras_layer?{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 188}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 188]}}
?

,kernel
-bias
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

.kernel
/bias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

0kernel
1bias
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 26, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 49}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
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
 :
??2dense/kernel
:?2
dense/bias
!:	?@2dense_1/kernel
:@2dense_1/bias
 :@ 2dense_2/kernel
: 2dense_2/bias
 : 2dense_3/kernel
:2dense_3/bias
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
0
1
2
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
%:#
??2Adam/dense/kernel/m
:?2Adam/dense/bias/m
&:$	?@2Adam/dense_1/kernel/m
:@2Adam/dense_1/bias/m
%:#@ 2Adam/dense_2/kernel/m
: 2Adam/dense_2/bias/m
%:# 2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
%:#
??2Adam/dense/kernel/v
:?2Adam/dense/bias/v
&:$	?@2Adam/dense_1/kernel/v
:@2Adam/dense_1/bias/v
%:#@ 2Adam/dense_2/kernel/v
: 2Adam/dense_2/bias/v
%:# 2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v
?2?
 __inference__wrapped_model_46990?
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
(
floor?
floor?????????
2

month_sold$?!

month_sold?????????
&
rent?
rent?????????
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
B__inference_model_1_layer_call_and_return_conditional_losses_48049
B__inference_model_1_layer_call_and_return_conditional_losses_48140
B__inference_model_1_layer_call_and_return_conditional_losses_47877
B__inference_model_1_layer_call_and_return_conditional_losses_47914?
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
'__inference_model_1_layer_call_fn_47695
'__inference_model_1_layer_call_fn_48176
'__inference_model_1_layer_call_fn_48212
'__inference_model_1_layer_call_fn_47840?
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
@__inference_model_layer_call_and_return_conditional_losses_48273
@__inference_model_layer_call_and_return_conditional_losses_48334
@__inference_model_layer_call_and_return_conditional_losses_47289
@__inference_model_layer_call_and_return_conditional_losses_47348?
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
%__inference_model_layer_call_fn_47092
%__inference_model_layer_call_fn_48354
%__inference_model_layer_call_fn_48374
%__inference_model_layer_call_fn_47230?
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
E__inference_sequential_layer_call_and_return_conditional_losses_48408
E__inference_sequential_layer_call_and_return_conditional_losses_48442
E__inference_sequential_layer_call_and_return_conditional_losses_47596
E__inference_sequential_layer_call_and_return_conditional_losses_47620?
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
*__inference_sequential_layer_call_fn_47445
*__inference_sequential_layer_call_fn_48463
*__inference_sequential_layer_call_fn_48484
*__inference_sequential_layer_call_fn_47572?
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
#__inference_signature_wrapper_47958areafloor
month_soldrentroomssquare_meteryear_of_construction	year_sold"?
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
F__inference_concatenate_layer_call_and_return_conditional_losses_48496?
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
+__inference_concatenate_layer_call_fn_48507?
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
__inference_adapt_step_48553?
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
H__inference_concatenate_1_layer_call_and_return_conditional_losses_48560?
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
-__inference_concatenate_1_layer_call_fn_48566?
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
@__inference_dense_layer_call_and_return_conditional_losses_48582?
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
%__inference_dense_layer_call_fn_48591?
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
B__inference_dense_1_layer_call_and_return_conditional_losses_48601?
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
'__inference_dense_1_layer_call_fn_48610?
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
B__inference_dense_2_layer_call_and_return_conditional_losses_48620?
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
'__inference_dense_2_layer_call_fn_48629?
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
B__inference_dense_3_layer_call_and_return_conditional_losses_48639?
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
'__inference_dense_3_layer_call_fn_48648?
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
__inference__creator_48653?
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
__inference__initializer_48658?
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
__inference__destroyer_48663?
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
__inference_save_fn_48682checkpoint_key"?
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
__inference_restore_fn_48690restored_tensors_0restored_tensors_1"?
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
__inference__creator_48653?

? 
? "? 8
__inference__destroyer_48663?

? 
? "? :
__inference__initializer_48658?

? 
? "? ?
 __inference__wrapped_model_46990?<?'(*+,-./01???
???
???
&
area?
area?????????
(
floor?
floor?????????
2

month_sold$?!

month_sold?????????
&
rent?
rent?????????
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
? "7?4
2

sequential$?!

sequential?????????l
__inference_adapt_step_48553L)'(A?>
7?4
2?/?
??????????IteratorSpec
? "
 ?
H__inference_concatenate_1_layer_call_and_return_conditional_losses_48560?[?X
Q?N
L?I
"?
inputs/0?????????
#? 
inputs/1??????????
? "&?#
?
0??????????
? ?
-__inference_concatenate_1_layer_call_fn_48566x[?X
Q?N
L?I
"?
inputs/0?????????
#? 
inputs/1??????????
? "????????????
F__inference_concatenate_layer_call_and_return_conditional_losses_48496????
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
+__inference_concatenate_layer_call_fn_48507????
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
B__inference_dense_1_layer_call_and_return_conditional_losses_48601],-0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? {
'__inference_dense_1_layer_call_fn_48610P,-0?-
&?#
!?
inputs??????????
? "??????????@?
B__inference_dense_2_layer_call_and_return_conditional_losses_48620\.//?,
%?"
 ?
inputs?????????@
? "%?"
?
0????????? 
? z
'__inference_dense_2_layer_call_fn_48629O.//?,
%?"
 ?
inputs?????????@
? "?????????? ?
B__inference_dense_3_layer_call_and_return_conditional_losses_48639\01/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? z
'__inference_dense_3_layer_call_fn_48648O01/?,
%?"
 ?
inputs????????? 
? "???????????
@__inference_dense_layer_call_and_return_conditional_losses_48582^*+0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? z
%__inference_dense_layer_call_fn_48591Q*+0?-
&?#
!?
inputs??????????
? "????????????
B__inference_model_1_layer_call_and_return_conditional_losses_47877?<?'(*+,-./01???
???
???
&
area?
area?????????
(
floor?
floor?????????
2

month_sold$?!

month_sold?????????
&
rent?
rent?????????
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
B__inference_model_1_layer_call_and_return_conditional_losses_47914?<?'(*+,-./01???
???
???
&
area?
area?????????
(
floor?
floor?????????
2

month_sold$?!

month_sold?????????
&
rent?
rent?????????
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
B__inference_model_1_layer_call_and_return_conditional_losses_48049?<?'(*+,-./01???
???
???
-
area%?"
inputs/area?????????
/
floor&?#
inputs/floor?????????
9

month_sold+?(
inputs/month_sold?????????
-
rent%?"
inputs/rent?????????
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
B__inference_model_1_layer_call_and_return_conditional_losses_48140?<?'(*+,-./01???
???
???
-
area%?"
inputs/area?????????
/
floor&?#
inputs/floor?????????
9

month_sold+?(
inputs/month_sold?????????
-
rent%?"
inputs/rent?????????
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
'__inference_model_1_layer_call_fn_47695?<?'(*+,-./01???
???
???
&
area?
area?????????
(
floor?
floor?????????
2

month_sold$?!

month_sold?????????
&
rent?
rent?????????
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
'__inference_model_1_layer_call_fn_47840?<?'(*+,-./01???
???
???
&
area?
area?????????
(
floor?
floor?????????
2

month_sold$?!

month_sold?????????
&
rent?
rent?????????
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
'__inference_model_1_layer_call_fn_48176?<?'(*+,-./01???
???
???
-
area%?"
inputs/area?????????
/
floor&?#
inputs/floor?????????
9

month_sold+?(
inputs/month_sold?????????
-
rent%?"
inputs/rent?????????
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
'__inference_model_1_layer_call_fn_48212?<?'(*+,-./01???
???
???
-
area%?"
inputs/area?????????
/
floor&?#
inputs/floor?????????
9

month_sold+?(
inputs/month_sold?????????
-
rent%?"
inputs/rent?????????
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
? "???????????
@__inference_model_layer_call_and_return_conditional_losses_47289?<?'(???
???
???
&
area?
area?????????
(
floor?
floor?????????
2

month_sold$?!

month_sold?????????
&
rent?
rent?????????
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
0??????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_47348?<?'(???
???
???
&
area?
area?????????
(
floor?
floor?????????
2

month_sold$?!

month_sold?????????
&
rent?
rent?????????
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
0??????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_48273?<?'(???
???
???
-
area%?"
inputs/area?????????
/
floor&?#
inputs/floor?????????
9

month_sold+?(
inputs/month_sold?????????
-
rent%?"
inputs/rent?????????
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
0??????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_48334?<?'(???
???
???
-
area%?"
inputs/area?????????
/
floor&?#
inputs/floor?????????
9

month_sold+?(
inputs/month_sold?????????
-
rent%?"
inputs/rent?????????
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
0??????????
? ?
%__inference_model_layer_call_fn_47092?<?'(???
???
???
&
area?
area?????????
(
floor?
floor?????????
2

month_sold$?!

month_sold?????????
&
rent?
rent?????????
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
? "????????????
%__inference_model_layer_call_fn_47230?<?'(???
???
???
&
area?
area?????????
(
floor?
floor?????????
2

month_sold$?!

month_sold?????????
&
rent?
rent?????????
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
? "????????????
%__inference_model_layer_call_fn_48354?<?'(???
???
???
-
area%?"
inputs/area?????????
/
floor&?#
inputs/floor?????????
9

month_sold+?(
inputs/month_sold?????????
-
rent%?"
inputs/rent?????????
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
? "????????????
%__inference_model_layer_call_fn_48374?<?'(???
???
???
-
area%?"
inputs/area?????????
/
floor&?#
inputs/floor?????????
9

month_sold+?(
inputs/month_sold?????????
-
rent%?"
inputs/rent?????????
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
? "???????????y
__inference_restore_fn_48690Y<K?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_48682?<&?#
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
E__inference_sequential_layer_call_and_return_conditional_losses_47596p*+,-./01=?:
3?0
&?#
dense_input??????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_47620p*+,-./01=?:
3?0
&?#
dense_input??????????
p

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_48408k*+,-./018?5
.?+
!?
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_48442k*+,-./018?5
.?+
!?
inputs??????????
p

 
? "%?"
?
0?????????
? ?
*__inference_sequential_layer_call_fn_47445c*+,-./01=?:
3?0
&?#
dense_input??????????
p 

 
? "???????????
*__inference_sequential_layer_call_fn_47572c*+,-./01=?:
3?0
&?#
dense_input??????????
p

 
? "???????????
*__inference_sequential_layer_call_fn_48463^*+,-./018?5
.?+
!?
inputs??????????
p 

 
? "???????????
*__inference_sequential_layer_call_fn_48484^*+,-./018?5
.?+
!?
inputs??????????
p

 
? "???????????
#__inference_signature_wrapper_47958?<?'(*+,-./01???
? 
???
&
area?
area?????????
(
floor?
floor?????????
2

month_sold$?!

month_sold?????????
&
rent?
rent?????????
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
	year_sold?????????"7?4
2

sequential$?!

sequential?????????