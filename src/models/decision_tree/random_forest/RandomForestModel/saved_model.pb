öę
çĘ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource
.
Identity

input"T
output"T"	
Ttype

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
ł
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
@
ReadVariableOp
resource
value"dtype"
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
f
SimpleMLCreateModelResource
model_handle"
	containerstring "
shared_namestring 
á
SimpleMLInferenceOpWithHandle
numerical_features
boolean_features
categorical_int_features'
#categorical_set_int_features_values1
-categorical_set_int_features_row_splits_dim_1	1
-categorical_set_int_features_row_splits_dim_2	
model_handle
dense_predictions
dense_col_representation"
dense_output_dimint(0
Ł
#SimpleMLLoadModelFromPathWithHandle
model_handle
path" 
output_typeslist(string)
 "
file_prefixstring " 
allow_slow_inferencebool(
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 
9
VarIsInitializedOp
resource
is_initialized
"serve*2.13.02unknown8Ăç
W
asset_path_initializerPlaceholder*
_output_shapes
: *
dtype0*
shape: 

VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
z
Variable/AssignAssignVariableOpVariableasset_path_initializer*&
 _has_manual_control_dependencies(*
dtype0
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
Y
asset_path_initializer_1Placeholder*
_output_shapes
: *
dtype0*
shape: 


Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 

Variable_1/AssignAssignVariableOp
Variable_1asset_path_initializer_1*&
 _has_manual_control_dependencies(*
dtype0
a
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
: *
dtype0
Y
asset_path_initializer_2Placeholder*
_output_shapes
: *
dtype0*
shape: 


Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 

Variable_2/AssignAssignVariableOp
Variable_2asset_path_initializer_2*&
 _has_manual_control_dependencies(*
dtype0
a
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
: *
dtype0
Y
asset_path_initializer_3Placeholder*
_output_shapes
: *
dtype0*
shape: 


Variable_3VarHandleOp*
_class
loc:@Variable_3*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_3
e
+Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_3*
_output_shapes
: 

Variable_3/AssignAssignVariableOp
Variable_3asset_path_initializer_3*&
 _has_manual_control_dependencies(*
dtype0
a
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes
: *
dtype0
Y
asset_path_initializer_4Placeholder*
_output_shapes
: *
dtype0*
shape: 


Variable_4VarHandleOp*
_class
loc:@Variable_4*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_4
e
+Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_4*
_output_shapes
: 

Variable_4/AssignAssignVariableOp
Variable_4asset_path_initializer_4*&
 _has_manual_control_dependencies(*
dtype0
a
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes
: *
dtype0
v
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
dtype0
t
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
v
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0

SimpleMLCreateModelResourceSimpleMLCreateModelResource*
_output_shapes
: *E
shared_name64simple_ml_model_6c8fd756-78dc-434f-a265-fcf1be0e696d
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
h

is_trainedVarHandleOp*
_output_shapes
: *
dtype0
*
shape: *
shared_name
is_trained
a
is_trained/Read/ReadVariableOpReadVariableOp
is_trained*
_output_shapes
: *
dtype0

m
serving_default_NoPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
w
serving_default_ack_flag_cntPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
u
serving_default_active_maxPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
v
serving_default_active_meanPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
u
serving_default_active_minPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
u
serving_default_active_stdPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
{
 serving_default_bwd_blk_rate_avgPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
y
serving_default_bwd_byts_b_avgPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
y
serving_default_bwd_header_lenPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
v
serving_default_bwd_iat_maxPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
w
serving_default_bwd_iat_meanPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
v
serving_default_bwd_iat_minPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
v
serving_default_bwd_iat_stdPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
v
serving_default_bwd_iat_totPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
z
serving_default_bwd_pkt_len_maxPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
{
 serving_default_bwd_pkt_len_meanPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
z
serving_default_bwd_pkt_len_minPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
z
serving_default_bwd_pkt_len_stdPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
y
serving_default_bwd_pkts_b_avgPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
u
serving_default_bwd_pkts_sPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
x
serving_default_bwd_psh_flagsPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
x
serving_default_bwd_urg_flagsPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
x
serving_default_down_up_ratioPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
q
serving_default_dst_ipPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙
s
serving_default_dst_portPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
w
serving_default_ece_flag_cntPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
w
serving_default_fin_flag_cntPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
v
serving_default_flow_byts_sPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
x
serving_default_flow_durationPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
w
serving_default_flow_iat_maxPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
x
serving_default_flow_iat_meanPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
w
serving_default_flow_iat_minPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
w
serving_default_flow_iat_stdPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
v
serving_default_flow_pkts_sPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
|
!serving_default_fwd_act_data_pktsPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
{
 serving_default_fwd_blk_rate_avgPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
y
serving_default_fwd_byts_b_avgPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
y
serving_default_fwd_header_lenPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
v
serving_default_fwd_iat_maxPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
w
serving_default_fwd_iat_meanPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
v
serving_default_fwd_iat_minPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
v
serving_default_fwd_iat_stdPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
v
serving_default_fwd_iat_totPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
z
serving_default_fwd_pkt_len_maxPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
{
 serving_default_fwd_pkt_len_meanPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
z
serving_default_fwd_pkt_len_minPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
z
serving_default_fwd_pkt_len_stdPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
y
serving_default_fwd_pkts_b_avgPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
u
serving_default_fwd_pkts_sPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
x
serving_default_fwd_psh_flagsPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
{
 serving_default_fwd_seg_size_minPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
x
serving_default_fwd_urg_flagsPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
s
serving_default_idle_maxPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
t
serving_default_idle_meanPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
s
serving_default_idle_minPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
s
serving_default_idle_stdPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
o
serving_default_infoPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
|
!serving_default_init_bwd_win_bytsPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
|
!serving_default_init_fwd_win_bytsPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
v
serving_default_pkt_len_maxPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
w
serving_default_pkt_len_meanPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
v
serving_default_pkt_len_minPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
v
serving_default_pkt_len_stdPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
v
serving_default_pkt_len_varPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
u
serving_default_pkt_lengthPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
w
serving_default_pkt_size_avgPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
s
serving_default_protocolPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
w
serving_default_psh_flag_cntPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
w
serving_default_rst_flag_cntPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
q
serving_default_src_ipPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙
s
serving_default_src_portPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
w
serving_default_syn_flag_cntPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
t
serving_default_timestampPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
w
serving_default_tot_bwd_pktsPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
w
serving_default_tot_fwd_pktsPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
z
serving_default_totlen_bwd_pktsPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
z
serving_default_totlen_fwd_pktsPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
w
serving_default_urg_flag_cntPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙

StatefulPartitionedCallStatefulPartitionedCallserving_default_Noserving_default_ack_flag_cntserving_default_active_maxserving_default_active_meanserving_default_active_minserving_default_active_std serving_default_bwd_blk_rate_avgserving_default_bwd_byts_b_avgserving_default_bwd_header_lenserving_default_bwd_iat_maxserving_default_bwd_iat_meanserving_default_bwd_iat_minserving_default_bwd_iat_stdserving_default_bwd_iat_totserving_default_bwd_pkt_len_max serving_default_bwd_pkt_len_meanserving_default_bwd_pkt_len_minserving_default_bwd_pkt_len_stdserving_default_bwd_pkts_b_avgserving_default_bwd_pkts_sserving_default_bwd_psh_flagsserving_default_bwd_urg_flagsserving_default_down_up_ratioserving_default_dst_ipserving_default_dst_portserving_default_ece_flag_cntserving_default_fin_flag_cntserving_default_flow_byts_sserving_default_flow_durationserving_default_flow_iat_maxserving_default_flow_iat_meanserving_default_flow_iat_minserving_default_flow_iat_stdserving_default_flow_pkts_s!serving_default_fwd_act_data_pkts serving_default_fwd_blk_rate_avgserving_default_fwd_byts_b_avgserving_default_fwd_header_lenserving_default_fwd_iat_maxserving_default_fwd_iat_meanserving_default_fwd_iat_minserving_default_fwd_iat_stdserving_default_fwd_iat_totserving_default_fwd_pkt_len_max serving_default_fwd_pkt_len_meanserving_default_fwd_pkt_len_minserving_default_fwd_pkt_len_stdserving_default_fwd_pkts_b_avgserving_default_fwd_pkts_sserving_default_fwd_psh_flags serving_default_fwd_seg_size_minserving_default_fwd_urg_flagsserving_default_idle_maxserving_default_idle_meanserving_default_idle_minserving_default_idle_stdserving_default_info!serving_default_init_bwd_win_byts!serving_default_init_fwd_win_bytsserving_default_pkt_len_maxserving_default_pkt_len_meanserving_default_pkt_len_minserving_default_pkt_len_stdserving_default_pkt_len_varserving_default_pkt_lengthserving_default_pkt_size_avgserving_default_protocolserving_default_psh_flag_cntserving_default_rst_flag_cntserving_default_src_ipserving_default_src_portserving_default_syn_flag_cntserving_default_timestampserving_default_tot_bwd_pktsserving_default_tot_fwd_pktsserving_default_totlen_bwd_pktsserving_default_totlen_fwd_pktsserving_default_urg_flag_cntSimpleMLCreateModelResource*Z
TinS
Q2O		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_signature_wrapper_4250
a
ReadVariableOpReadVariableOpVariable^Variable/Assign*
_output_shapes
: *
dtype0
Ú
StatefulPartitionedCall_1StatefulPartitionedCallReadVariableOpSimpleMLCreateModelResource*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *&
f!R
__inference__initializer_4261

NoOpNoOp^StatefulPartitionedCall_1^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign
Ú
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB B
Ą
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

_multitask
	_is_trained

_learner_params
	_features
	optimizer
loss
_models
_build_normalized_inputs
_finalize_predictions
call
call_get_leaves
yggdrasil_model_path_tensor

signatures*

	0*
* 
* 
°
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
* 
JD
VARIABLE_VALUE
is_trained&_is_trained/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
O

_variables
_iterations
 _learning_rate
!_update_step_xla*
* 
	
"0* 

#trace_0* 

$trace_0* 

%trace_0* 
* 

&trace_0* 

'serving_default* 

	0*
* 
 
(0
)1
*2
+3*
* 
* 
* 
* 
* 
* 

0*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
+
,_input_builder
-_compiled_model* 
* 
* 
* 

.	capture_0* 
* 
8
/	variables
0	keras_api
	1total
	2count*
H
3	variables
4	keras_api
	5total
	6count
7
_fn_kwargs*
[
8	variables
9	keras_api
:
thresholds
;true_positives
<false_positives*
[
=	variables
>	keras_api
?
thresholds
@true_positives
Afalse_negatives*
P
B_feature_name_to_idx
C	_init_ops
#Dcategorical_str_to_int_hashmaps* 
S
E_model_loader
F_create_resource
G_initialize
H_destroy_resource* 
* 

10
21*

/	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

50
61*

3	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

;0
<1*

8	variables*
* 
ga
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

@0
A1*

=	variables*
* 
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_negatives>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
5
I_output_types
J
_all_files
.
_done_file* 

Ktrace_0* 

Ltrace_0* 

Mtrace_0* 
* 
%
N0
O1
.2
P3
Q4* 
* 

.	capture_0* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ą
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename
is_trained	iterationlearning_ratetotal_1count_1totalcounttrue_positives_1false_positivestrue_positivesfalse_negativesConst*
Tin
2*
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
GPU2*0J 8 *&
f!R
__inference__traced_save_4453
Ź
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filename
is_trained	iterationlearning_ratetotal_1count_1totalcounttrue_positives_1false_positivestrue_positivesfalse_negatives*
Tin
2*
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
GPU2*0J 8 *)
f$R"
 __inference__traced_restore_4495Ţş

šV
í
__inference__wrapped_model_3315
no
ack_flag_cnt

active_max
active_mean

active_min

active_std
bwd_blk_rate_avg
bwd_byts_b_avg
bwd_header_len
bwd_iat_max
bwd_iat_mean
bwd_iat_min
bwd_iat_std
bwd_iat_tot
bwd_pkt_len_max
bwd_pkt_len_mean
bwd_pkt_len_min
bwd_pkt_len_std
bwd_pkts_b_avg

bwd_pkts_s
bwd_psh_flags
bwd_urg_flags
down_up_ratio

dst_ip	
dst_port
ece_flag_cnt
fin_flag_cnt
flow_byts_s
flow_duration
flow_iat_max
flow_iat_mean
flow_iat_min
flow_iat_std
flow_pkts_s
fwd_act_data_pkts
fwd_blk_rate_avg
fwd_byts_b_avg
fwd_header_len
fwd_iat_max
fwd_iat_mean
fwd_iat_min
fwd_iat_std
fwd_iat_tot
fwd_pkt_len_max
fwd_pkt_len_mean
fwd_pkt_len_min
fwd_pkt_len_std
fwd_pkts_b_avg

fwd_pkts_s
fwd_psh_flags
fwd_seg_size_min
fwd_urg_flags
idle_max
	idle_mean
idle_min
idle_std
info
init_bwd_win_byts
init_fwd_win_byts
pkt_len_max
pkt_len_mean
pkt_len_min
pkt_len_std
pkt_len_var

pkt_length
pkt_size_avg
protocol
psh_flag_cnt
rst_flag_cnt

src_ip	
src_port
syn_flag_cnt
	timestamp
tot_bwd_pkts
tot_fwd_pkts
totlen_bwd_pkts
totlen_fwd_pkts
urg_flag_cnt
randomforestmodel_3311
identity˘)RandomForestModel/StatefulPartitionedCallž
)RandomForestModel/StatefulPartitionedCallStatefulPartitionedCallnoack_flag_cnt
active_maxactive_mean
active_min
active_stdbwd_blk_rate_avgbwd_byts_b_avgbwd_header_lenbwd_iat_maxbwd_iat_meanbwd_iat_minbwd_iat_stdbwd_iat_totbwd_pkt_len_maxbwd_pkt_len_meanbwd_pkt_len_minbwd_pkt_len_stdbwd_pkts_b_avg
bwd_pkts_sbwd_psh_flagsbwd_urg_flagsdown_up_ratiodst_ipdst_portece_flag_cntfin_flag_cntflow_byts_sflow_durationflow_iat_maxflow_iat_meanflow_iat_minflow_iat_stdflow_pkts_sfwd_act_data_pktsfwd_blk_rate_avgfwd_byts_b_avgfwd_header_lenfwd_iat_maxfwd_iat_meanfwd_iat_minfwd_iat_stdfwd_iat_totfwd_pkt_len_maxfwd_pkt_len_meanfwd_pkt_len_minfwd_pkt_len_stdfwd_pkts_b_avg
fwd_pkts_sfwd_psh_flagsfwd_seg_size_minfwd_urg_flagsidle_max	idle_meanidle_minidle_stdinfoinit_bwd_win_bytsinit_fwd_win_bytspkt_len_maxpkt_len_meanpkt_len_minpkt_len_stdpkt_len_var
pkt_lengthpkt_size_avgprotocolpsh_flag_cntrst_flag_cntsrc_ipsrc_portsyn_flag_cnt	timestamptot_bwd_pktstot_fwd_pktstotlen_bwd_pktstotlen_fwd_pktsurg_flag_cntrandomforestmodel_3311*Z
TinS
Q2O		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *
fR
__inference_call_3310
IdentityIdentity2RandomForestModel/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙N
NoOpNoOp*^RandomForestModel/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Š	
_input_shapes	
	:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: 2V
)RandomForestModel/StatefulPartitionedCall)RandomForestModel/StatefulPartitionedCall:$N 

_user_specified_name3311:QMM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameurg_flag_cnt:TLP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nametotlen_fwd_pkts:TKP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nametotlen_bwd_pkts:QJM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nametot_fwd_pkts:QIM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nametot_bwd_pkts:NHJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	timestamp:QGM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namesyn_flag_cnt:MFI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
src_port:KEG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namesrc_ip:QDM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namerst_flag_cnt:QCM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namepsh_flag_cnt:MBI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
protocol:QAM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namepkt_size_avg:O@K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
pkt_length:P?L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepkt_len_var:P>L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepkt_len_std:P=L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepkt_len_min:Q<M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namepkt_len_mean:P;L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepkt_len_max:V:R
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinit_fwd_win_byts:V9R
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinit_bwd_win_byts:I8E
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinfo:M7I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
idle_std:M6I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
idle_min:N5J
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	idle_mean:M4I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
idle_max:R3N
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namefwd_urg_flags:U2Q
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namefwd_seg_size_min:R1N
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namefwd_psh_flags:O0K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
fwd_pkts_s:S/O
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namefwd_pkts_b_avg:T.P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefwd_pkt_len_std:T-P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefwd_pkt_len_min:U,Q
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namefwd_pkt_len_mean:T+P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefwd_pkt_len_max:P*L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefwd_iat_tot:P)L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefwd_iat_std:P(L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefwd_iat_min:Q'M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefwd_iat_mean:P&L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefwd_iat_max:S%O
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namefwd_header_len:S$O
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namefwd_byts_b_avg:U#Q
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namefwd_blk_rate_avg:V"R
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_namefwd_act_data_pkts:P!L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameflow_pkts_s:Q M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameflow_iat_std:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameflow_iat_min:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameflow_iat_mean:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameflow_iat_max:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameflow_duration:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameflow_byts_s:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefin_flag_cnt:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameece_flag_cnt:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
dst_port:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namedst_ip:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namedown_up_ratio:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namebwd_urg_flags:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namebwd_psh_flags:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
bwd_pkts_s:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namebwd_pkts_b_avg:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namebwd_pkt_len_std:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namebwd_pkt_len_min:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namebwd_pkt_len_mean:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namebwd_pkt_len_max:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namebwd_iat_tot:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namebwd_iat_std:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namebwd_iat_min:Q
M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namebwd_iat_mean:P	L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namebwd_iat_max:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namebwd_header_len:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namebwd_byts_b_avg:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namebwd_blk_rate_avg:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
active_std:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
active_min:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameactive_mean:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
active_max:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameack_flag_cnt:G C
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNo
˛
ž
__inference__initializer_4261
staticregexreplace_input>
:simple_ml_simplemlloadmodelfrompathwithhandle_model_handle
identity˘-simple_ml/SimpleMLLoadModelFromPathWithHandle
StaticRegexReplaceStaticRegexReplacestaticregexreplace_input*
_output_shapes
: *!
pattern7d524b07892540dadone*
rewrite ć
-simple_ml/SimpleMLLoadModelFromPathWithHandle#SimpleMLLoadModelFromPathWithHandle:simple_ml_simplemlloadmodelfrompathwithhandle_model_handleStaticRegexReplace:output:0*
_output_shapes
 *!
file_prefix7d524b07892540daG
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: R
NoOpNoOp.^simple_ml/SimpleMLLoadModelFromPathWithHandle*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2^
-simple_ml/SimpleMLLoadModelFromPathWithHandle-simple_ml/SimpleMLLoadModelFromPathWithHandle:,(
&
_user_specified_namemodel_handle: 

_output_shapes
: 


)__inference__build_normalized_inputs_3215

inputs
	inputs_57
	inputs_64
	inputs_66
	inputs_65
	inputs_67
	inputs_77
	inputs_74
	inputs_32
	inputs_45
	inputs_47
	inputs_46
	inputs_48
	inputs_44
	inputs_22
	inputs_24
	inputs_23
	inputs_25
	inputs_75
	inputs_13
	inputs_50
	inputs_52
	inputs_60
inputs_2	
inputs_4
	inputs_59
	inputs_53
	inputs_10
inputs_9
	inputs_36
	inputs_35
	inputs_37
	inputs_38
	inputs_11
	inputs_34
	inputs_76
	inputs_72
	inputs_31
	inputs_40
	inputs_42
	inputs_41
	inputs_43
	inputs_39
	inputs_18
	inputs_20
	inputs_19
	inputs_21
	inputs_73
	inputs_12
	inputs_49
	inputs_33
	inputs_51
	inputs_68
	inputs_70
	inputs_69
	inputs_71
inputs_7
	inputs_63
	inputs_62
	inputs_26
	inputs_28
	inputs_27
	inputs_29
	inputs_30
inputs_6
	inputs_61
inputs_5
	inputs_56
	inputs_55
inputs_1	
inputs_3
	inputs_54
inputs_8
	inputs_15
	inputs_14
	inputs_17
	inputs_16
	inputs_58
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11
identity_12
identity_13
identity_14
identity_15
identity_16
identity_17
identity_18
identity_19
identity_20
identity_21
identity_22
identity_23
identity_24
identity_25
identity_26
identity_27
identity_28
identity_29
identity_30
identity_31
identity_32
identity_33
identity_34
identity_35
identity_36
identity_37
identity_38
identity_39
identity_40
identity_41
identity_42
identity_43
identity_44
identity_45
identity_46
identity_47
identity_48
identity_49
identity_50
identity_51
identity_52
identity_53
identity_54
identity_55
identity_56
identity_57
identity_58
identity_59
identity_60
identity_61
identity_62
identity_63
identity_64
identity_65
identity_66
identity_67
identity_68
identity_69
identity_70
identity_71
identity_72
identity_73S
CastCastinputs_3*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙U
Cast_1Castinputs_4*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙U
Cast_2Castinputs_5*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙U
Cast_3Castinputs_6*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙V
Cast_4Cast	inputs_31*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙V
Cast_5Cast	inputs_32*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙V
Cast_6Cast	inputs_49*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙V
Cast_7Cast	inputs_50*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙V
Cast_8Cast	inputs_51*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙V
Cast_9Cast	inputs_52*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙W
Cast_10Cast	inputs_53*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙W
Cast_11Cast	inputs_54*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙W
Cast_12Cast	inputs_55*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙W
Cast_13Cast	inputs_56*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙W
Cast_14Cast	inputs_57*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙W
Cast_15Cast	inputs_58*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙W
Cast_16Cast	inputs_59*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O
IdentityIdentityCast_14:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O

Identity_1Identity	inputs_64*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O

Identity_2Identity	inputs_66*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O

Identity_3Identity	inputs_65*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O

Identity_4Identity	inputs_67*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O

Identity_5Identity	inputs_77*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O

Identity_6Identity	inputs_74*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P

Identity_7Identity
Cast_5:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O

Identity_8Identity	inputs_45*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O

Identity_9Identity	inputs_47*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_10Identity	inputs_46*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_11Identity	inputs_48*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_12Identity	inputs_44*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_13Identity	inputs_22*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_14Identity	inputs_24*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_15Identity	inputs_23*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_16Identity	inputs_25*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_17Identity	inputs_75*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_18Identity	inputs_13*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
Identity_19Identity
Cast_7:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
Identity_20Identity
Cast_9:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_21Identity	inputs_60*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
Identity_22Identity
Cast_1:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙R
Identity_23IdentityCast_16:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙R
Identity_24IdentityCast_10:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_25Identity	inputs_10*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O
Identity_26Identityinputs_9*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_27Identity	inputs_36*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_28Identity	inputs_35*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_29Identity	inputs_37*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_30Identity	inputs_38*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_31Identity	inputs_11*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_32Identity	inputs_34*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_33Identity	inputs_76*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_34Identity	inputs_72*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
Identity_35Identity
Cast_4:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_36Identity	inputs_40*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_37Identity	inputs_42*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_38Identity	inputs_41*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_39Identity	inputs_43*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_40Identity	inputs_39*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_41Identity	inputs_18*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_42Identity	inputs_20*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_43Identity	inputs_19*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_44Identity	inputs_21*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_45Identity	inputs_73*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_46Identity	inputs_12*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
Identity_47Identity
Cast_6:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_48Identity	inputs_33*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
Identity_49Identity
Cast_8:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_50Identity	inputs_68*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_51Identity	inputs_70*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_52Identity	inputs_69*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_53Identity	inputs_71*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O
Identity_54Identityinputs_7*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_55Identity	inputs_63*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_56Identity	inputs_62*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_57Identity	inputs_26*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_58Identity	inputs_28*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_59Identity	inputs_27*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_60Identity	inputs_29*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_61Identity	inputs_30*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
Identity_62Identity
Cast_3:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_63Identity	inputs_61*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
Identity_64Identity
Cast_2:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙R
Identity_65IdentityCast_13:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙R
Identity_66IdentityCast_12:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O
Identity_67IdentityCast:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙R
Identity_68IdentityCast_11:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_69Identity	inputs_15*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_70Identity	inputs_14*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_71Identity	inputs_17*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P
Identity_72Identity	inputs_16*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙R
Identity_73IdentityCast_15:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0"#
identity_14Identity_14:output:0"#
identity_15Identity_15:output:0"#
identity_16Identity_16:output:0"#
identity_17Identity_17:output:0"#
identity_18Identity_18:output:0"#
identity_19Identity_19:output:0"!

identity_1Identity_1:output:0"#
identity_20Identity_20:output:0"#
identity_21Identity_21:output:0"#
identity_22Identity_22:output:0"#
identity_23Identity_23:output:0"#
identity_24Identity_24:output:0"#
identity_25Identity_25:output:0"#
identity_26Identity_26:output:0"#
identity_27Identity_27:output:0"#
identity_28Identity_28:output:0"#
identity_29Identity_29:output:0"!

identity_2Identity_2:output:0"#
identity_30Identity_30:output:0"#
identity_31Identity_31:output:0"#
identity_32Identity_32:output:0"#
identity_33Identity_33:output:0"#
identity_34Identity_34:output:0"#
identity_35Identity_35:output:0"#
identity_36Identity_36:output:0"#
identity_37Identity_37:output:0"#
identity_38Identity_38:output:0"#
identity_39Identity_39:output:0"!

identity_3Identity_3:output:0"#
identity_40Identity_40:output:0"#
identity_41Identity_41:output:0"#
identity_42Identity_42:output:0"#
identity_43Identity_43:output:0"#
identity_44Identity_44:output:0"#
identity_45Identity_45:output:0"#
identity_46Identity_46:output:0"#
identity_47Identity_47:output:0"#
identity_48Identity_48:output:0"#
identity_49Identity_49:output:0"!

identity_4Identity_4:output:0"#
identity_50Identity_50:output:0"#
identity_51Identity_51:output:0"#
identity_52Identity_52:output:0"#
identity_53Identity_53:output:0"#
identity_54Identity_54:output:0"#
identity_55Identity_55:output:0"#
identity_56Identity_56:output:0"#
identity_57Identity_57:output:0"#
identity_58Identity_58:output:0"#
identity_59Identity_59:output:0"!

identity_5Identity_5:output:0"#
identity_60Identity_60:output:0"#
identity_61Identity_61:output:0"#
identity_62Identity_62:output:0"#
identity_63Identity_63:output:0"#
identity_64Identity_64:output:0"#
identity_65Identity_65:output:0"#
identity_66Identity_66:output:0"#
identity_67Identity_67:output:0"#
identity_68Identity_68:output:0"#
identity_69Identity_69:output:0"!

identity_6Identity_6:output:0"#
identity_70Identity_70:output:0"#
identity_71Identity_71:output:0"#
identity_72Identity_72:output:0"#
identity_73Identity_73:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*§	
_input_shapes	
	:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:KMG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KLG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KKG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KJG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KIG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KHG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KGG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KFG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KEG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KDG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KCG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KBG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KAG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K@G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K?G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K>G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K=G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K<G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K;G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K:G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K9G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K8G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K7G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K6G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K5G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K4G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K3G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K2G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K1G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K0G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K/G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K.G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K-G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K,G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K+G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K*G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K)G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K(G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K'G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K&G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K%G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K$G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K#G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K"G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K!G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K
G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K	G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

+
__inference__destroyer_4265
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
őU
Ý
0__inference_RandomForestModel_layer_call_fn_3729
no
ack_flag_cnt

active_max
active_mean

active_min

active_std
bwd_blk_rate_avg
bwd_byts_b_avg
bwd_header_len
bwd_iat_max
bwd_iat_mean
bwd_iat_min
bwd_iat_std
bwd_iat_tot
bwd_pkt_len_max
bwd_pkt_len_mean
bwd_pkt_len_min
bwd_pkt_len_std
bwd_pkts_b_avg

bwd_pkts_s
bwd_psh_flags
bwd_urg_flags
down_up_ratio

dst_ip	
dst_port
ece_flag_cnt
fin_flag_cnt
flow_byts_s
flow_duration
flow_iat_max
flow_iat_mean
flow_iat_min
flow_iat_std
flow_pkts_s
fwd_act_data_pkts
fwd_blk_rate_avg
fwd_byts_b_avg
fwd_header_len
fwd_iat_max
fwd_iat_mean
fwd_iat_min
fwd_iat_std
fwd_iat_tot
fwd_pkt_len_max
fwd_pkt_len_mean
fwd_pkt_len_min
fwd_pkt_len_std
fwd_pkts_b_avg

fwd_pkts_s
fwd_psh_flags
fwd_seg_size_min
fwd_urg_flags
idle_max
	idle_mean
idle_min
idle_std
info
init_bwd_win_byts
init_fwd_win_byts
pkt_len_max
pkt_len_mean
pkt_len_min
pkt_len_std
pkt_len_var

pkt_length
pkt_size_avg
protocol
psh_flag_cnt
rst_flag_cnt

src_ip	
src_port
syn_flag_cnt
	timestamp
tot_bwd_pkts
tot_fwd_pkts
totlen_bwd_pkts
totlen_fwd_pkts
urg_flag_cnt
unknown
identity˘StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallnoack_flag_cnt
active_maxactive_mean
active_min
active_stdbwd_blk_rate_avgbwd_byts_b_avgbwd_header_lenbwd_iat_maxbwd_iat_meanbwd_iat_minbwd_iat_stdbwd_iat_totbwd_pkt_len_maxbwd_pkt_len_meanbwd_pkt_len_minbwd_pkt_len_stdbwd_pkts_b_avg
bwd_pkts_sbwd_psh_flagsbwd_urg_flagsdown_up_ratiodst_ipdst_portece_flag_cntfin_flag_cntflow_byts_sflow_durationflow_iat_maxflow_iat_meanflow_iat_minflow_iat_stdflow_pkts_sfwd_act_data_pktsfwd_blk_rate_avgfwd_byts_b_avgfwd_header_lenfwd_iat_maxfwd_iat_meanfwd_iat_minfwd_iat_stdfwd_iat_totfwd_pkt_len_maxfwd_pkt_len_meanfwd_pkt_len_minfwd_pkt_len_stdfwd_pkts_b_avg
fwd_pkts_sfwd_psh_flagsfwd_seg_size_minfwd_urg_flagsidle_max	idle_meanidle_minidle_stdinfoinit_bwd_win_bytsinit_fwd_win_bytspkt_len_maxpkt_len_meanpkt_len_minpkt_len_stdpkt_len_var
pkt_lengthpkt_size_avgprotocolpsh_flag_cntrst_flag_cntsrc_ipsrc_portsyn_flag_cnt	timestamptot_bwd_pktstot_fwd_pktstotlen_bwd_pktstotlen_fwd_pktsurg_flag_cntunknown*Z
TinS
Q2O		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_RandomForestModel_layer_call_and_return_conditional_losses_3480o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Š	
_input_shapes	
	:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: 22
StatefulPartitionedCallStatefulPartitionedCall:$N 

_user_specified_name3725:QMM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameurg_flag_cnt:TLP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nametotlen_fwd_pkts:TKP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nametotlen_bwd_pkts:QJM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nametot_fwd_pkts:QIM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nametot_bwd_pkts:NHJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	timestamp:QGM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namesyn_flag_cnt:MFI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
src_port:KEG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namesrc_ip:QDM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namerst_flag_cnt:QCM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namepsh_flag_cnt:MBI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
protocol:QAM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namepkt_size_avg:O@K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
pkt_length:P?L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepkt_len_var:P>L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepkt_len_std:P=L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepkt_len_min:Q<M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namepkt_len_mean:P;L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepkt_len_max:V:R
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinit_fwd_win_byts:V9R
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinit_bwd_win_byts:I8E
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinfo:M7I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
idle_std:M6I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
idle_min:N5J
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	idle_mean:M4I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
idle_max:R3N
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namefwd_urg_flags:U2Q
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namefwd_seg_size_min:R1N
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namefwd_psh_flags:O0K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
fwd_pkts_s:S/O
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namefwd_pkts_b_avg:T.P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefwd_pkt_len_std:T-P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefwd_pkt_len_min:U,Q
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namefwd_pkt_len_mean:T+P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefwd_pkt_len_max:P*L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefwd_iat_tot:P)L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefwd_iat_std:P(L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefwd_iat_min:Q'M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefwd_iat_mean:P&L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefwd_iat_max:S%O
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namefwd_header_len:S$O
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namefwd_byts_b_avg:U#Q
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namefwd_blk_rate_avg:V"R
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_namefwd_act_data_pkts:P!L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameflow_pkts_s:Q M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameflow_iat_std:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameflow_iat_min:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameflow_iat_mean:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameflow_iat_max:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameflow_duration:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameflow_byts_s:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefin_flag_cnt:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameece_flag_cnt:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
dst_port:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namedst_ip:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namedown_up_ratio:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namebwd_urg_flags:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namebwd_psh_flags:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
bwd_pkts_s:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namebwd_pkts_b_avg:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namebwd_pkt_len_std:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namebwd_pkt_len_min:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namebwd_pkt_len_mean:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namebwd_pkt_len_max:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namebwd_iat_tot:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namebwd_iat_std:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namebwd_iat_min:Q
M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namebwd_iat_mean:P	L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namebwd_iat_max:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namebwd_header_len:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namebwd_byts_b_avg:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namebwd_blk_rate_avg:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
active_std:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
active_min:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameactive_mean:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
active_max:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameack_flag_cnt:G C
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNo
Ű

&__inference__finalize_predictions_3995!
predictions_dense_predictions(
$predictions_dense_col_representation
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ű
strided_sliceStridedSlicepredictions_dense_predictionsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

begin_mask*
end_mask^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:˙˙˙˙˙˙˙˙˙::`\

_output_shapes
:
>
_user_specified_name&$predictions_dense_col_representation:f b
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namepredictions_dense_predictions
\
´	
__inference__traced_save_4453
file_prefix+
!read_disablecopyonread_is_trained:
 ,
"read_1_disablecopyonread_iteration:	 0
&read_2_disablecopyonread_learning_rate: *
 read_3_disablecopyonread_total_1: *
 read_4_disablecopyonread_count_1: (
read_5_disablecopyonread_total: (
read_6_disablecopyonread_count: 7
)read_7_disablecopyonread_true_positives_1:6
(read_8_disablecopyonread_false_positives:5
'read_9_disablecopyonread_true_positives:7
)read_10_disablecopyonread_false_negatives:
savev2_const
identity_23˘MergeV2Checkpoints˘Read/DisableCopyOnRead˘Read/ReadVariableOp˘Read_1/DisableCopyOnRead˘Read_1/ReadVariableOp˘Read_10/DisableCopyOnRead˘Read_10/ReadVariableOp˘Read_2/DisableCopyOnRead˘Read_2/ReadVariableOp˘Read_3/DisableCopyOnRead˘Read_3/ReadVariableOp˘Read_4/DisableCopyOnRead˘Read_4/ReadVariableOp˘Read_5/DisableCopyOnRead˘Read_5/ReadVariableOp˘Read_6/DisableCopyOnRead˘Read_6/ReadVariableOp˘Read_7/DisableCopyOnRead˘Read_7/ReadVariableOp˘Read_8/DisableCopyOnRead˘Read_8/ReadVariableOp˘Read_9/DisableCopyOnRead˘Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: s
Read/DisableCopyOnReadDisableCopyOnRead!read_disablecopyonread_is_trained"/device:CPU:0*
_output_shapes
 
Read/ReadVariableOpReadVariableOp!read_disablecopyonread_is_trained^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0
a
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0
*
_output_shapes
: Y

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0
*
_output_shapes
: v
Read_1/DisableCopyOnReadDisableCopyOnRead"read_1_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 
Read_1/ReadVariableOpReadVariableOp"read_1_disablecopyonread_iteration^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	e

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: [

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_2/DisableCopyOnReadDisableCopyOnRead&read_2_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 
Read_2/ReadVariableOpReadVariableOp&read_2_disablecopyonread_learning_rate^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0e

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: [

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_3/DisableCopyOnReadDisableCopyOnRead read_3_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 
Read_3/ReadVariableOpReadVariableOp read_3_disablecopyonread_total_1^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0e

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: [

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_4/DisableCopyOnReadDisableCopyOnRead read_4_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 
Read_4/ReadVariableOpReadVariableOp read_4_disablecopyonread_count_1^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0e

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: [

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
: r
Read_5/DisableCopyOnReadDisableCopyOnReadread_5_disablecopyonread_total"/device:CPU:0*
_output_shapes
 
Read_5/ReadVariableOpReadVariableOpread_5_disablecopyonread_total^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: r
Read_6/DisableCopyOnReadDisableCopyOnReadread_6_disablecopyonread_count"/device:CPU:0*
_output_shapes
 
Read_6/ReadVariableOpReadVariableOpread_6_disablecopyonread_count^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
: }
Read_7/DisableCopyOnReadDisableCopyOnRead)read_7_disablecopyonread_true_positives_1"/device:CPU:0*
_output_shapes
 Ľ
Read_7/ReadVariableOpReadVariableOp)read_7_disablecopyonread_true_positives_1^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_8/DisableCopyOnReadDisableCopyOnRead(read_8_disablecopyonread_false_positives"/device:CPU:0*
_output_shapes
 ¤
Read_8/ReadVariableOpReadVariableOp(read_8_disablecopyonread_false_positives^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:{
Read_9/DisableCopyOnReadDisableCopyOnRead'read_9_disablecopyonread_true_positives"/device:CPU:0*
_output_shapes
 Ł
Read_9/ReadVariableOpReadVariableOp'read_9_disablecopyonread_true_positives^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_10/DisableCopyOnReadDisableCopyOnRead)read_10_disablecopyonread_false_negatives"/device:CPU:0*
_output_shapes
 §
Read_10/ReadVariableOpReadVariableOp)read_10_disablecopyonread_false_negatives^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:ď
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB&_is_trained/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B Đ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2
	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:ł
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_22Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_23IdentityIdentity_22:output:0^NoOp*
T0*
_output_shapes
: ć
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_23Identity_23:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:/+
)
_user_specified_namefalse_negatives:.
*
(
_user_specified_nametrue_positives:/	+
)
_user_specified_namefalse_positives:0,
*
_user_specified_nametrue_positives_1:%!

_user_specified_namecount:%!

_user_specified_nametotal:'#
!
_user_specified_name	count_1:'#
!
_user_specified_name	total_1:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:*&
$
_user_specified_name
is_trained:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ź
J
__inference__creator_4254
identity˘SimpleMLCreateModelResource
SimpleMLCreateModelResourceSimpleMLCreateModelResource*
_output_shapes
: *E
shared_name64simple_ml_model_6c8fd756-78dc-434f-a265-fcf1be0e696dh
IdentityIdentity*SimpleMLCreateModelResource:model_handle:0^NoOp*
T0*
_output_shapes
: @
NoOpNoOp^SimpleMLCreateModelResource*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2:
SimpleMLCreateModelResourceSimpleMLCreateModelResource
ťU
Ď
"__inference_signature_wrapper_4250
no
ack_flag_cnt

active_max
active_mean

active_min

active_std
bwd_blk_rate_avg
bwd_byts_b_avg
bwd_header_len
bwd_iat_max
bwd_iat_mean
bwd_iat_min
bwd_iat_std
bwd_iat_tot
bwd_pkt_len_max
bwd_pkt_len_mean
bwd_pkt_len_min
bwd_pkt_len_std
bwd_pkts_b_avg

bwd_pkts_s
bwd_psh_flags
bwd_urg_flags
down_up_ratio

dst_ip	
dst_port
ece_flag_cnt
fin_flag_cnt
flow_byts_s
flow_duration
flow_iat_max
flow_iat_mean
flow_iat_min
flow_iat_std
flow_pkts_s
fwd_act_data_pkts
fwd_blk_rate_avg
fwd_byts_b_avg
fwd_header_len
fwd_iat_max
fwd_iat_mean
fwd_iat_min
fwd_iat_std
fwd_iat_tot
fwd_pkt_len_max
fwd_pkt_len_mean
fwd_pkt_len_min
fwd_pkt_len_std
fwd_pkts_b_avg

fwd_pkts_s
fwd_psh_flags
fwd_seg_size_min
fwd_urg_flags
idle_max
	idle_mean
idle_min
idle_std
info
init_bwd_win_byts
init_fwd_win_byts
pkt_len_max
pkt_len_mean
pkt_len_min
pkt_len_std
pkt_len_var

pkt_length
pkt_size_avg
protocol
psh_flag_cnt
rst_flag_cnt

src_ip	
src_port
syn_flag_cnt
	timestamp
tot_bwd_pkts
tot_fwd_pkts
totlen_bwd_pkts
totlen_fwd_pkts
urg_flag_cnt
unknown
identity˘StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallnoack_flag_cnt
active_maxactive_mean
active_min
active_stdbwd_blk_rate_avgbwd_byts_b_avgbwd_header_lenbwd_iat_maxbwd_iat_meanbwd_iat_minbwd_iat_stdbwd_iat_totbwd_pkt_len_maxbwd_pkt_len_meanbwd_pkt_len_minbwd_pkt_len_stdbwd_pkts_b_avg
bwd_pkts_sbwd_psh_flagsbwd_urg_flagsdown_up_ratiodst_ipdst_portece_flag_cntfin_flag_cntflow_byts_sflow_durationflow_iat_maxflow_iat_meanflow_iat_minflow_iat_stdflow_pkts_sfwd_act_data_pktsfwd_blk_rate_avgfwd_byts_b_avgfwd_header_lenfwd_iat_maxfwd_iat_meanfwd_iat_minfwd_iat_stdfwd_iat_totfwd_pkt_len_maxfwd_pkt_len_meanfwd_pkt_len_minfwd_pkt_len_stdfwd_pkts_b_avg
fwd_pkts_sfwd_psh_flagsfwd_seg_size_minfwd_urg_flagsidle_max	idle_meanidle_minidle_stdinfoinit_bwd_win_bytsinit_fwd_win_bytspkt_len_maxpkt_len_meanpkt_len_minpkt_len_stdpkt_len_var
pkt_lengthpkt_size_avgprotocolpsh_flag_cntrst_flag_cntsrc_ipsrc_portsyn_flag_cnt	timestamptot_bwd_pktstot_fwd_pktstotlen_bwd_pktstotlen_fwd_pktsurg_flag_cntunknown*Z
TinS
Q2O		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__wrapped_model_3315o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Š	
_input_shapes	
	:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: 22
StatefulPartitionedCallStatefulPartitionedCall:$N 

_user_specified_name4246:QMM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameurg_flag_cnt:TLP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nametotlen_fwd_pkts:TKP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nametotlen_bwd_pkts:QJM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nametot_fwd_pkts:QIM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nametot_bwd_pkts:NHJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	timestamp:QGM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namesyn_flag_cnt:MFI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
src_port:KEG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namesrc_ip:QDM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namerst_flag_cnt:QCM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namepsh_flag_cnt:MBI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
protocol:QAM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namepkt_size_avg:O@K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
pkt_length:P?L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepkt_len_var:P>L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepkt_len_std:P=L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepkt_len_min:Q<M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namepkt_len_mean:P;L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepkt_len_max:V:R
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinit_fwd_win_byts:V9R
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinit_bwd_win_byts:I8E
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinfo:M7I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
idle_std:M6I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
idle_min:N5J
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	idle_mean:M4I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
idle_max:R3N
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namefwd_urg_flags:U2Q
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namefwd_seg_size_min:R1N
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namefwd_psh_flags:O0K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
fwd_pkts_s:S/O
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namefwd_pkts_b_avg:T.P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefwd_pkt_len_std:T-P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefwd_pkt_len_min:U,Q
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namefwd_pkt_len_mean:T+P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefwd_pkt_len_max:P*L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefwd_iat_tot:P)L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefwd_iat_std:P(L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefwd_iat_min:Q'M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefwd_iat_mean:P&L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefwd_iat_max:S%O
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namefwd_header_len:S$O
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namefwd_byts_b_avg:U#Q
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namefwd_blk_rate_avg:V"R
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_namefwd_act_data_pkts:P!L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameflow_pkts_s:Q M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameflow_iat_std:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameflow_iat_min:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameflow_iat_mean:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameflow_iat_max:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameflow_duration:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameflow_byts_s:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefin_flag_cnt:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameece_flag_cnt:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
dst_port:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namedst_ip:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namedown_up_ratio:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namebwd_urg_flags:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namebwd_psh_flags:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
bwd_pkts_s:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namebwd_pkts_b_avg:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namebwd_pkt_len_std:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namebwd_pkt_len_min:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namebwd_pkt_len_mean:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namebwd_pkt_len_max:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namebwd_iat_tot:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namebwd_iat_std:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namebwd_iat_min:Q
M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namebwd_iat_mean:P	L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namebwd_iat_max:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namebwd_header_len:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namebwd_byts_b_avg:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namebwd_blk_rate_avg:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
active_std:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
active_min:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameactive_mean:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
active_max:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameack_flag_cnt:G C
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNo
Ąw
˙
K__inference_RandomForestModel_layer_call_and_return_conditional_losses_3480
no
ack_flag_cnt

active_max
active_mean

active_min

active_std
bwd_blk_rate_avg
bwd_byts_b_avg
bwd_header_len
bwd_iat_max
bwd_iat_mean
bwd_iat_min
bwd_iat_std
bwd_iat_tot
bwd_pkt_len_max
bwd_pkt_len_mean
bwd_pkt_len_min
bwd_pkt_len_std
bwd_pkts_b_avg

bwd_pkts_s
bwd_psh_flags
bwd_urg_flags
down_up_ratio

dst_ip	
dst_port
ece_flag_cnt
fin_flag_cnt
flow_byts_s
flow_duration
flow_iat_max
flow_iat_mean
flow_iat_min
flow_iat_std
flow_pkts_s
fwd_act_data_pkts
fwd_blk_rate_avg
fwd_byts_b_avg
fwd_header_len
fwd_iat_max
fwd_iat_mean
fwd_iat_min
fwd_iat_std
fwd_iat_tot
fwd_pkt_len_max
fwd_pkt_len_mean
fwd_pkt_len_min
fwd_pkt_len_std
fwd_pkts_b_avg

fwd_pkts_s
fwd_psh_flags
fwd_seg_size_min
fwd_urg_flags
idle_max
	idle_mean
idle_min
idle_std
info
init_bwd_win_byts
init_fwd_win_byts
pkt_len_max
pkt_len_mean
pkt_len_min
pkt_len_std
pkt_len_var

pkt_length
pkt_size_avg
protocol
psh_flag_cnt
rst_flag_cnt

src_ip	
src_port
syn_flag_cnt
	timestamp
tot_bwd_pkts
tot_fwd_pkts
totlen_bwd_pkts
totlen_fwd_pkts
urg_flag_cnt
inference_op_model_handle
identity˘inference_opŚ
PartitionedCallPartitionedCallnoack_flag_cnt
active_maxactive_mean
active_min
active_stdbwd_blk_rate_avgbwd_byts_b_avgbwd_header_lenbwd_iat_maxbwd_iat_meanbwd_iat_minbwd_iat_stdbwd_iat_totbwd_pkt_len_maxbwd_pkt_len_meanbwd_pkt_len_minbwd_pkt_len_stdbwd_pkts_b_avg
bwd_pkts_sbwd_psh_flagsbwd_urg_flagsdown_up_ratiodst_ipdst_portece_flag_cntfin_flag_cntflow_byts_sflow_durationflow_iat_maxflow_iat_meanflow_iat_minflow_iat_stdflow_pkts_sfwd_act_data_pktsfwd_blk_rate_avgfwd_byts_b_avgfwd_header_lenfwd_iat_maxfwd_iat_meanfwd_iat_minfwd_iat_stdfwd_iat_totfwd_pkt_len_maxfwd_pkt_len_meanfwd_pkt_len_minfwd_pkt_len_stdfwd_pkts_b_avg
fwd_pkts_sfwd_psh_flagsfwd_seg_size_minfwd_urg_flagsidle_max	idle_meanidle_minidle_stdinfoinit_bwd_win_bytsinit_fwd_win_bytspkt_len_maxpkt_len_meanpkt_len_minpkt_len_stdpkt_len_var
pkt_lengthpkt_size_avgprotocolpsh_flag_cntrst_flag_cntsrc_ipsrc_portsyn_flag_cnt	timestamptot_bwd_pktstot_fwd_pktstotlen_bwd_pktstotlen_fwd_pktsurg_flag_cnt*Y
TinR
P2N		*V
ToutN
L2J*
_collective_manager_ids
 *ě
_output_shapesŮ
Ö:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference__build_normalized_inputs_3215
stackPackPartitionedCall:output:0PartitionedCall:output:1PartitionedCall:output:2PartitionedCall:output:3PartitionedCall:output:4PartitionedCall:output:5PartitionedCall:output:6PartitionedCall:output:7PartitionedCall:output:8PartitionedCall:output:9PartitionedCall:output:10PartitionedCall:output:11PartitionedCall:output:12PartitionedCall:output:13PartitionedCall:output:14PartitionedCall:output:15PartitionedCall:output:16PartitionedCall:output:17PartitionedCall:output:18PartitionedCall:output:19PartitionedCall:output:20PartitionedCall:output:21PartitionedCall:output:22PartitionedCall:output:23PartitionedCall:output:24PartitionedCall:output:25PartitionedCall:output:26PartitionedCall:output:27PartitionedCall:output:28PartitionedCall:output:29PartitionedCall:output:30PartitionedCall:output:31PartitionedCall:output:32PartitionedCall:output:33PartitionedCall:output:34PartitionedCall:output:35PartitionedCall:output:36PartitionedCall:output:37PartitionedCall:output:38PartitionedCall:output:39PartitionedCall:output:40PartitionedCall:output:41PartitionedCall:output:42PartitionedCall:output:43PartitionedCall:output:44PartitionedCall:output:45PartitionedCall:output:46PartitionedCall:output:47PartitionedCall:output:48PartitionedCall:output:49PartitionedCall:output:50PartitionedCall:output:51PartitionedCall:output:52PartitionedCall:output:53PartitionedCall:output:54PartitionedCall:output:55PartitionedCall:output:56PartitionedCall:output:57PartitionedCall:output:58PartitionedCall:output:59PartitionedCall:output:60PartitionedCall:output:61PartitionedCall:output:62PartitionedCall:output:63PartitionedCall:output:64PartitionedCall:output:65PartitionedCall:output:66PartitionedCall:output:67PartitionedCall:output:68PartitionedCall:output:69PartitionedCall:output:70PartitionedCall:output:71PartitionedCall:output:72PartitionedCall:output:73*
NJ*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙J*

axisL
ConstConst*
_output_shapes
:  *
dtype0*
value
B  N
Const_1Const*
_output_shapes
:  *
dtype0*
value
B  X
RaggedConstant/valuesConst*
_output_shapes
: *
dtype0*
valueB ^
RaggedConstant/ConstConst*
_output_shapes
:*
dtype0	*
valueB	R `
RaggedConstant/Const_1Const*
_output_shapes
:*
dtype0	*
valueB	R Ą
inference_opSimpleMLInferenceOpWithHandlestack:output:0Const:output:0Const_1:output:0RaggedConstant/values:output:0RaggedConstant/Const:output:0RaggedConstant/Const_1:output:0inference_op_model_handle*-
_output_shapes
:˙˙˙˙˙˙˙˙˙:*
dense_output_dimÚ
PartitionedCall_1PartitionedCall inference_op:dense_predictions:0'inference_op:dense_col_representation:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 */
f*R(
&__inference__finalize_predictions_3307i
IdentityIdentityPartitionedCall_1:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙1
NoOpNoOp^inference_op*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Š	
_input_shapes	
	:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: 2
inference_opinference_op:,N(
&
_user_specified_namemodel_handle:QMM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameurg_flag_cnt:TLP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nametotlen_fwd_pkts:TKP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nametotlen_bwd_pkts:QJM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nametot_fwd_pkts:QIM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nametot_bwd_pkts:NHJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	timestamp:QGM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namesyn_flag_cnt:MFI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
src_port:KEG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namesrc_ip:QDM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namerst_flag_cnt:QCM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namepsh_flag_cnt:MBI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
protocol:QAM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namepkt_size_avg:O@K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
pkt_length:P?L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepkt_len_var:P>L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepkt_len_std:P=L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepkt_len_min:Q<M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namepkt_len_mean:P;L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepkt_len_max:V:R
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinit_fwd_win_byts:V9R
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinit_bwd_win_byts:I8E
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinfo:M7I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
idle_std:M6I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
idle_min:N5J
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	idle_mean:M4I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
idle_max:R3N
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namefwd_urg_flags:U2Q
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namefwd_seg_size_min:R1N
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namefwd_psh_flags:O0K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
fwd_pkts_s:S/O
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namefwd_pkts_b_avg:T.P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefwd_pkt_len_std:T-P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefwd_pkt_len_min:U,Q
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namefwd_pkt_len_mean:T+P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefwd_pkt_len_max:P*L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefwd_iat_tot:P)L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefwd_iat_std:P(L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefwd_iat_min:Q'M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefwd_iat_mean:P&L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefwd_iat_max:S%O
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namefwd_header_len:S$O
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namefwd_byts_b_avg:U#Q
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namefwd_blk_rate_avg:V"R
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_namefwd_act_data_pkts:P!L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameflow_pkts_s:Q M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameflow_iat_std:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameflow_iat_min:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameflow_iat_mean:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameflow_iat_max:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameflow_duration:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameflow_byts_s:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefin_flag_cnt:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameece_flag_cnt:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
dst_port:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namedst_ip:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namedown_up_ratio:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namebwd_urg_flags:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namebwd_psh_flags:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
bwd_pkts_s:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namebwd_pkts_b_avg:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namebwd_pkt_len_std:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namebwd_pkt_len_min:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namebwd_pkt_len_mean:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namebwd_pkt_len_max:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namebwd_iat_tot:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namebwd_iat_std:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namebwd_iat_min:Q
M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namebwd_iat_mean:P	L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namebwd_iat_max:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namebwd_header_len:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namebwd_byts_b_avg:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namebwd_blk_rate_avg:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
active_std:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
active_min:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameactive_mean:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
active_max:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameack_flag_cnt:G C
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNo
Ń
ë
__inference_call_4160
	inputs_no
inputs_ack_flag_cnt
inputs_active_max
inputs_active_mean
inputs_active_min
inputs_active_std
inputs_bwd_blk_rate_avg
inputs_bwd_byts_b_avg
inputs_bwd_header_len
inputs_bwd_iat_max
inputs_bwd_iat_mean
inputs_bwd_iat_min
inputs_bwd_iat_std
inputs_bwd_iat_tot
inputs_bwd_pkt_len_max
inputs_bwd_pkt_len_mean
inputs_bwd_pkt_len_min
inputs_bwd_pkt_len_std
inputs_bwd_pkts_b_avg
inputs_bwd_pkts_s
inputs_bwd_psh_flags
inputs_bwd_urg_flags
inputs_down_up_ratio
inputs_dst_ip	
inputs_dst_port
inputs_ece_flag_cnt
inputs_fin_flag_cnt
inputs_flow_byts_s
inputs_flow_duration
inputs_flow_iat_max
inputs_flow_iat_mean
inputs_flow_iat_min
inputs_flow_iat_std
inputs_flow_pkts_s
inputs_fwd_act_data_pkts
inputs_fwd_blk_rate_avg
inputs_fwd_byts_b_avg
inputs_fwd_header_len
inputs_fwd_iat_max
inputs_fwd_iat_mean
inputs_fwd_iat_min
inputs_fwd_iat_std
inputs_fwd_iat_tot
inputs_fwd_pkt_len_max
inputs_fwd_pkt_len_mean
inputs_fwd_pkt_len_min
inputs_fwd_pkt_len_std
inputs_fwd_pkts_b_avg
inputs_fwd_pkts_s
inputs_fwd_psh_flags
inputs_fwd_seg_size_min
inputs_fwd_urg_flags
inputs_idle_max
inputs_idle_mean
inputs_idle_min
inputs_idle_std
inputs_info
inputs_init_bwd_win_byts
inputs_init_fwd_win_byts
inputs_pkt_len_max
inputs_pkt_len_mean
inputs_pkt_len_min
inputs_pkt_len_std
inputs_pkt_len_var
inputs_pkt_length
inputs_pkt_size_avg
inputs_protocol
inputs_psh_flag_cnt
inputs_rst_flag_cnt
inputs_src_ip	
inputs_src_port
inputs_syn_flag_cnt
inputs_timestamp
inputs_tot_bwd_pkts
inputs_tot_fwd_pkts
inputs_totlen_bwd_pkts
inputs_totlen_fwd_pkts
inputs_urg_flag_cnt
inference_op_model_handle
identity˘inference_opČ
PartitionedCallPartitionedCall	inputs_noinputs_ack_flag_cntinputs_active_maxinputs_active_meaninputs_active_mininputs_active_stdinputs_bwd_blk_rate_avginputs_bwd_byts_b_avginputs_bwd_header_leninputs_bwd_iat_maxinputs_bwd_iat_meaninputs_bwd_iat_mininputs_bwd_iat_stdinputs_bwd_iat_totinputs_bwd_pkt_len_maxinputs_bwd_pkt_len_meaninputs_bwd_pkt_len_mininputs_bwd_pkt_len_stdinputs_bwd_pkts_b_avginputs_bwd_pkts_sinputs_bwd_psh_flagsinputs_bwd_urg_flagsinputs_down_up_ratioinputs_dst_ipinputs_dst_portinputs_ece_flag_cntinputs_fin_flag_cntinputs_flow_byts_sinputs_flow_durationinputs_flow_iat_maxinputs_flow_iat_meaninputs_flow_iat_mininputs_flow_iat_stdinputs_flow_pkts_sinputs_fwd_act_data_pktsinputs_fwd_blk_rate_avginputs_fwd_byts_b_avginputs_fwd_header_leninputs_fwd_iat_maxinputs_fwd_iat_meaninputs_fwd_iat_mininputs_fwd_iat_stdinputs_fwd_iat_totinputs_fwd_pkt_len_maxinputs_fwd_pkt_len_meaninputs_fwd_pkt_len_mininputs_fwd_pkt_len_stdinputs_fwd_pkts_b_avginputs_fwd_pkts_sinputs_fwd_psh_flagsinputs_fwd_seg_size_mininputs_fwd_urg_flagsinputs_idle_maxinputs_idle_meaninputs_idle_mininputs_idle_stdinputs_infoinputs_init_bwd_win_bytsinputs_init_fwd_win_bytsinputs_pkt_len_maxinputs_pkt_len_meaninputs_pkt_len_mininputs_pkt_len_stdinputs_pkt_len_varinputs_pkt_lengthinputs_pkt_size_avginputs_protocolinputs_psh_flag_cntinputs_rst_flag_cntinputs_src_ipinputs_src_portinputs_syn_flag_cntinputs_timestampinputs_tot_bwd_pktsinputs_tot_fwd_pktsinputs_totlen_bwd_pktsinputs_totlen_fwd_pktsinputs_urg_flag_cnt*Y
TinR
P2N		*V
ToutN
L2J*
_collective_manager_ids
 *ě
_output_shapesŮ
Ö:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference__build_normalized_inputs_3215
stackPackPartitionedCall:output:0PartitionedCall:output:1PartitionedCall:output:2PartitionedCall:output:3PartitionedCall:output:4PartitionedCall:output:5PartitionedCall:output:6PartitionedCall:output:7PartitionedCall:output:8PartitionedCall:output:9PartitionedCall:output:10PartitionedCall:output:11PartitionedCall:output:12PartitionedCall:output:13PartitionedCall:output:14PartitionedCall:output:15PartitionedCall:output:16PartitionedCall:output:17PartitionedCall:output:18PartitionedCall:output:19PartitionedCall:output:20PartitionedCall:output:21PartitionedCall:output:22PartitionedCall:output:23PartitionedCall:output:24PartitionedCall:output:25PartitionedCall:output:26PartitionedCall:output:27PartitionedCall:output:28PartitionedCall:output:29PartitionedCall:output:30PartitionedCall:output:31PartitionedCall:output:32PartitionedCall:output:33PartitionedCall:output:34PartitionedCall:output:35PartitionedCall:output:36PartitionedCall:output:37PartitionedCall:output:38PartitionedCall:output:39PartitionedCall:output:40PartitionedCall:output:41PartitionedCall:output:42PartitionedCall:output:43PartitionedCall:output:44PartitionedCall:output:45PartitionedCall:output:46PartitionedCall:output:47PartitionedCall:output:48PartitionedCall:output:49PartitionedCall:output:50PartitionedCall:output:51PartitionedCall:output:52PartitionedCall:output:53PartitionedCall:output:54PartitionedCall:output:55PartitionedCall:output:56PartitionedCall:output:57PartitionedCall:output:58PartitionedCall:output:59PartitionedCall:output:60PartitionedCall:output:61PartitionedCall:output:62PartitionedCall:output:63PartitionedCall:output:64PartitionedCall:output:65PartitionedCall:output:66PartitionedCall:output:67PartitionedCall:output:68PartitionedCall:output:69PartitionedCall:output:70PartitionedCall:output:71PartitionedCall:output:72PartitionedCall:output:73*
NJ*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙J*

axisL
ConstConst*
_output_shapes
:  *
dtype0*
value
B  N
Const_1Const*
_output_shapes
:  *
dtype0*
value
B  X
RaggedConstant/valuesConst*
_output_shapes
: *
dtype0*
valueB ^
RaggedConstant/ConstConst*
_output_shapes
:*
dtype0	*
valueB	R `
RaggedConstant/Const_1Const*
_output_shapes
:*
dtype0	*
valueB	R Ą
inference_opSimpleMLInferenceOpWithHandlestack:output:0Const:output:0Const_1:output:0RaggedConstant/values:output:0RaggedConstant/Const:output:0RaggedConstant/Const_1:output:0inference_op_model_handle*-
_output_shapes
:˙˙˙˙˙˙˙˙˙:*
dense_output_dimÚ
PartitionedCall_1PartitionedCall inference_op:dense_predictions:0'inference_op:dense_col_representation:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 */
f*R(
&__inference__finalize_predictions_3307i
IdentityIdentityPartitionedCall_1:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙1
NoOpNoOp^inference_op*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Š	
_input_shapes	
	:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: 2
inference_opinference_op:,N(
&
_user_specified_namemodel_handle:XMT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_urg_flag_cnt:[LW
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs_totlen_fwd_pkts:[KW
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs_totlen_bwd_pkts:XJT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_tot_fwd_pkts:XIT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_tot_bwd_pkts:UHQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameinputs_timestamp:XGT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_syn_flag_cnt:TFP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameinputs_src_port:REN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameinputs_src_ip:XDT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_rst_flag_cnt:XCT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_psh_flag_cnt:TBP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameinputs_protocol:XAT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_pkt_size_avg:V@R
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_pkt_length:W?S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_pkt_len_var:W>S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_pkt_len_std:W=S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_pkt_len_min:X<T
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_pkt_len_mean:W;S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_pkt_len_max:]:Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_nameinputs_init_fwd_win_byts:]9Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_nameinputs_init_bwd_win_byts:P8L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameinputs_info:T7P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameinputs_idle_std:T6P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameinputs_idle_min:U5Q
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameinputs_idle_mean:T4P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameinputs_idle_max:Y3U
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_fwd_urg_flags:\2X
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_nameinputs_fwd_seg_size_min:Y1U
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_fwd_psh_flags:V0R
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_fwd_pkts_s:Z/V
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs_fwd_pkts_b_avg:[.W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs_fwd_pkt_len_std:[-W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs_fwd_pkt_len_min:\,X
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_nameinputs_fwd_pkt_len_mean:[+W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs_fwd_pkt_len_max:W*S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_fwd_iat_tot:W)S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_fwd_iat_std:W(S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_fwd_iat_min:X'T
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_fwd_iat_mean:W&S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_fwd_iat_max:Z%V
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs_fwd_header_len:Z$V
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs_fwd_byts_b_avg:\#X
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_nameinputs_fwd_blk_rate_avg:]"Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_nameinputs_fwd_act_data_pkts:W!S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_flow_pkts_s:X T
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_flow_iat_std:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_flow_iat_min:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_flow_iat_mean:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_flow_iat_max:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_flow_duration:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_flow_byts_s:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_fin_flag_cnt:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_ece_flag_cnt:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameinputs_dst_port:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameinputs_dst_ip:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_down_up_ratio:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_bwd_urg_flags:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_bwd_psh_flags:VR
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_bwd_pkts_s:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs_bwd_pkts_b_avg:[W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs_bwd_pkt_len_std:[W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs_bwd_pkt_len_min:\X
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_nameinputs_bwd_pkt_len_mean:[W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs_bwd_pkt_len_max:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_bwd_iat_tot:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_bwd_iat_std:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_bwd_iat_min:X
T
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_bwd_iat_mean:W	S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_bwd_iat_max:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs_bwd_header_len:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs_bwd_byts_b_avg:\X
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_nameinputs_bwd_blk_rate_avg:VR
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_active_std:VR
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_active_min:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_active_mean:VR
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_active_max:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_ack_flag_cnt:N J
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_no
ô
Z
&__inference__finalize_predictions_3307
predictions
predictions_1
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      é
strided_sliceStridedSlicepredictionsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

begin_mask*
end_mask^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:˙˙˙˙˙˙˙˙˙::GC

_output_shapes
:
%
_user_specified_namepredictions:T P
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepredictions
Ö5

 __inference__traced_restore_4495
file_prefix%
assignvariableop_is_trained:
 &
assignvariableop_1_iteration:	 *
 assignvariableop_2_learning_rate: $
assignvariableop_3_total_1: $
assignvariableop_4_count_1: "
assignvariableop_5_total: "
assignvariableop_6_count: 1
#assignvariableop_7_true_positives_1:0
"assignvariableop_8_false_positives:/
!assignvariableop_9_true_positives:1
#assignvariableop_10_false_negatives:
identity_12˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_2˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9ň
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB&_is_trained/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B Ú
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*D
_output_shapes2
0::::::::::::*
dtypes
2
	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0
*
_output_shapes
:Ž
AssignVariableOpAssignVariableOpassignvariableop_is_trainedIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0
]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0	*
_output_shapes
:ł
AssignVariableOp_1AssignVariableOpassignvariableop_1_iterationIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:ˇ
AssignVariableOp_2AssignVariableOp assignvariableop_2_learning_rateIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:ą
AssignVariableOp_3AssignVariableOpassignvariableop_3_total_1Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:ą
AssignVariableOp_4AssignVariableOpassignvariableop_4_count_1Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ż
AssignVariableOp_5AssignVariableOpassignvariableop_5_totalIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ż
AssignVariableOp_6AssignVariableOpassignvariableop_6_countIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:ş
AssignVariableOp_7AssignVariableOp#assignvariableop_7_true_positives_1Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:š
AssignVariableOp_8AssignVariableOp"assignvariableop_8_false_positivesIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_9AssignVariableOp!assignvariableop_9_true_positivesIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:ź
AssignVariableOp_10AssignVariableOp#assignvariableop_10_false_negativesIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 Á
Identity_11Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_12IdentityIdentity_11:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_12Identity_12:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
: : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:/+
)
_user_specified_namefalse_negatives:.
*
(
_user_specified_nametrue_positives:/	+
)
_user_specified_namefalse_positives:0,
*
_user_specified_nametrue_positives_1:%!

_user_specified_namecount:%!

_user_specified_nametotal:'#
!
_user_specified_name	count_1:'#
!
_user_specified_name	total_1:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:*&
$
_user_specified_name
is_trained:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ćo
Ü	
__inference_call_3310

inputs
	inputs_57
	inputs_64
	inputs_66
	inputs_65
	inputs_67
	inputs_77
	inputs_74
	inputs_32
	inputs_45
	inputs_47
	inputs_46
	inputs_48
	inputs_44
	inputs_22
	inputs_24
	inputs_23
	inputs_25
	inputs_75
	inputs_13
	inputs_50
	inputs_52
	inputs_60
inputs_2	
inputs_4
	inputs_59
	inputs_53
	inputs_10
inputs_9
	inputs_36
	inputs_35
	inputs_37
	inputs_38
	inputs_11
	inputs_34
	inputs_76
	inputs_72
	inputs_31
	inputs_40
	inputs_42
	inputs_41
	inputs_43
	inputs_39
	inputs_18
	inputs_20
	inputs_19
	inputs_21
	inputs_73
	inputs_12
	inputs_49
	inputs_33
	inputs_51
	inputs_68
	inputs_70
	inputs_69
	inputs_71
inputs_7
	inputs_63
	inputs_62
	inputs_26
	inputs_28
	inputs_27
	inputs_29
	inputs_30
inputs_6
	inputs_61
inputs_5
	inputs_56
	inputs_55
inputs_1	
inputs_3
	inputs_54
inputs_8
	inputs_15
	inputs_14
	inputs_17
	inputs_16
	inputs_58
inference_op_model_handle
identity˘inference_opš
PartitionedCallPartitionedCallinputs	inputs_57	inputs_64	inputs_66	inputs_65	inputs_67	inputs_77	inputs_74	inputs_32	inputs_45	inputs_47	inputs_46	inputs_48	inputs_44	inputs_22	inputs_24	inputs_23	inputs_25	inputs_75	inputs_13	inputs_50	inputs_52	inputs_60inputs_2inputs_4	inputs_59	inputs_53	inputs_10inputs_9	inputs_36	inputs_35	inputs_37	inputs_38	inputs_11	inputs_34	inputs_76	inputs_72	inputs_31	inputs_40	inputs_42	inputs_41	inputs_43	inputs_39	inputs_18	inputs_20	inputs_19	inputs_21	inputs_73	inputs_12	inputs_49	inputs_33	inputs_51	inputs_68	inputs_70	inputs_69	inputs_71inputs_7	inputs_63	inputs_62	inputs_26	inputs_28	inputs_27	inputs_29	inputs_30inputs_6	inputs_61inputs_5	inputs_56	inputs_55inputs_1inputs_3	inputs_54inputs_8	inputs_15	inputs_14	inputs_17	inputs_16	inputs_58*Y
TinR
P2N		*V
ToutN
L2J*
_collective_manager_ids
 *ě
_output_shapesŮ
Ö:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference__build_normalized_inputs_3215
stackPackPartitionedCall:output:0PartitionedCall:output:1PartitionedCall:output:2PartitionedCall:output:3PartitionedCall:output:4PartitionedCall:output:5PartitionedCall:output:6PartitionedCall:output:7PartitionedCall:output:8PartitionedCall:output:9PartitionedCall:output:10PartitionedCall:output:11PartitionedCall:output:12PartitionedCall:output:13PartitionedCall:output:14PartitionedCall:output:15PartitionedCall:output:16PartitionedCall:output:17PartitionedCall:output:18PartitionedCall:output:19PartitionedCall:output:20PartitionedCall:output:21PartitionedCall:output:22PartitionedCall:output:23PartitionedCall:output:24PartitionedCall:output:25PartitionedCall:output:26PartitionedCall:output:27PartitionedCall:output:28PartitionedCall:output:29PartitionedCall:output:30PartitionedCall:output:31PartitionedCall:output:32PartitionedCall:output:33PartitionedCall:output:34PartitionedCall:output:35PartitionedCall:output:36PartitionedCall:output:37PartitionedCall:output:38PartitionedCall:output:39PartitionedCall:output:40PartitionedCall:output:41PartitionedCall:output:42PartitionedCall:output:43PartitionedCall:output:44PartitionedCall:output:45PartitionedCall:output:46PartitionedCall:output:47PartitionedCall:output:48PartitionedCall:output:49PartitionedCall:output:50PartitionedCall:output:51PartitionedCall:output:52PartitionedCall:output:53PartitionedCall:output:54PartitionedCall:output:55PartitionedCall:output:56PartitionedCall:output:57PartitionedCall:output:58PartitionedCall:output:59PartitionedCall:output:60PartitionedCall:output:61PartitionedCall:output:62PartitionedCall:output:63PartitionedCall:output:64PartitionedCall:output:65PartitionedCall:output:66PartitionedCall:output:67PartitionedCall:output:68PartitionedCall:output:69PartitionedCall:output:70PartitionedCall:output:71PartitionedCall:output:72PartitionedCall:output:73*
NJ*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙J*

axisL
ConstConst*
_output_shapes
:  *
dtype0*
value
B  N
Const_1Const*
_output_shapes
:  *
dtype0*
value
B  X
RaggedConstant/valuesConst*
_output_shapes
: *
dtype0*
valueB ^
RaggedConstant/ConstConst*
_output_shapes
:*
dtype0	*
valueB	R `
RaggedConstant/Const_1Const*
_output_shapes
:*
dtype0	*
valueB	R Ą
inference_opSimpleMLInferenceOpWithHandlestack:output:0Const:output:0Const_1:output:0RaggedConstant/values:output:0RaggedConstant/Const:output:0RaggedConstant/Const_1:output:0inference_op_model_handle*-
_output_shapes
:˙˙˙˙˙˙˙˙˙:*
dense_output_dimÚ
PartitionedCall_1PartitionedCall inference_op:dense_predictions:0'inference_op:dense_col_representation:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 */
f*R(
&__inference__finalize_predictions_3307i
IdentityIdentityPartitionedCall_1:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙1
NoOpNoOp^inference_op*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Š	
_input_shapes	
	:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: 2
inference_opinference_op:,N(
&
_user_specified_namemodel_handle:KMG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KLG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KKG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KJG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KIG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KHG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KGG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KFG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KEG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KDG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KCG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KBG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KAG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K@G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K?G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K>G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K=G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K<G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K;G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K:G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K9G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K8G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K7G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K6G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K5G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K4G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K3G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K2G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K1G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K0G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K/G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K.G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K-G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K,G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K+G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K*G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K)G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K(G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K'G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K&G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K%G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K$G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K#G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K"G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K!G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K
G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K	G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
°

)__inference__build_normalized_inputs_3986
	inputs_no
inputs_ack_flag_cnt
inputs_active_max
inputs_active_mean
inputs_active_min
inputs_active_std
inputs_bwd_blk_rate_avg
inputs_bwd_byts_b_avg
inputs_bwd_header_len
inputs_bwd_iat_max
inputs_bwd_iat_mean
inputs_bwd_iat_min
inputs_bwd_iat_std
inputs_bwd_iat_tot
inputs_bwd_pkt_len_max
inputs_bwd_pkt_len_mean
inputs_bwd_pkt_len_min
inputs_bwd_pkt_len_std
inputs_bwd_pkts_b_avg
inputs_bwd_pkts_s
inputs_bwd_psh_flags
inputs_bwd_urg_flags
inputs_down_up_ratio
inputs_dst_ip	
inputs_dst_port
inputs_ece_flag_cnt
inputs_fin_flag_cnt
inputs_flow_byts_s
inputs_flow_duration
inputs_flow_iat_max
inputs_flow_iat_mean
inputs_flow_iat_min
inputs_flow_iat_std
inputs_flow_pkts_s
inputs_fwd_act_data_pkts
inputs_fwd_blk_rate_avg
inputs_fwd_byts_b_avg
inputs_fwd_header_len
inputs_fwd_iat_max
inputs_fwd_iat_mean
inputs_fwd_iat_min
inputs_fwd_iat_std
inputs_fwd_iat_tot
inputs_fwd_pkt_len_max
inputs_fwd_pkt_len_mean
inputs_fwd_pkt_len_min
inputs_fwd_pkt_len_std
inputs_fwd_pkts_b_avg
inputs_fwd_pkts_s
inputs_fwd_psh_flags
inputs_fwd_seg_size_min
inputs_fwd_urg_flags
inputs_idle_max
inputs_idle_mean
inputs_idle_min
inputs_idle_std
inputs_info
inputs_init_bwd_win_byts
inputs_init_fwd_win_byts
inputs_pkt_len_max
inputs_pkt_len_mean
inputs_pkt_len_min
inputs_pkt_len_std
inputs_pkt_len_var
inputs_pkt_length
inputs_pkt_size_avg
inputs_protocol
inputs_psh_flag_cnt
inputs_rst_flag_cnt
inputs_src_ip	
inputs_src_port
inputs_syn_flag_cnt
inputs_timestamp
inputs_tot_bwd_pkts
inputs_tot_fwd_pkts
inputs_totlen_bwd_pkts
inputs_totlen_fwd_pkts
inputs_urg_flag_cnt
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11
identity_12
identity_13
identity_14
identity_15
identity_16
identity_17
identity_18
identity_19
identity_20
identity_21
identity_22
identity_23
identity_24
identity_25
identity_26
identity_27
identity_28
identity_29
identity_30
identity_31
identity_32
identity_33
identity_34
identity_35
identity_36
identity_37
identity_38
identity_39
identity_40
identity_41
identity_42
identity_43
identity_44
identity_45
identity_46
identity_47
identity_48
identity_49
identity_50
identity_51
identity_52
identity_53
identity_54
identity_55
identity_56
identity_57
identity_58
identity_59
identity_60
identity_61
identity_62
identity_63
identity_64
identity_65
identity_66
identity_67
identity_68
identity_69
identity_70
identity_71
identity_72
identity_73Z
CastCastinputs_src_port*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙\
Cast_1Castinputs_dst_port*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙\
Cast_2Castinputs_protocol*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙^
Cast_3Castinputs_pkt_length*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙b
Cast_4Castinputs_fwd_header_len*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙b
Cast_5Castinputs_bwd_header_len*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙a
Cast_6Castinputs_fwd_psh_flags*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙a
Cast_7Castinputs_bwd_psh_flags*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙a
Cast_8Castinputs_fwd_urg_flags*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙a
Cast_9Castinputs_bwd_urg_flags*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙a
Cast_10Castinputs_fin_flag_cnt*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙a
Cast_11Castinputs_syn_flag_cnt*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙a
Cast_12Castinputs_rst_flag_cnt*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙a
Cast_13Castinputs_psh_flag_cnt*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙a
Cast_14Castinputs_ack_flag_cnt*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙a
Cast_15Castinputs_urg_flag_cnt*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙a
Cast_16Castinputs_ece_flag_cnt*

DstT0*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O
IdentityIdentityCast_14:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙W

Identity_1Identityinputs_active_max*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙X

Identity_2Identityinputs_active_mean*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙W

Identity_3Identityinputs_active_min*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙W

Identity_4Identityinputs_active_std*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙]

Identity_5Identityinputs_bwd_blk_rate_avg*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_6Identityinputs_bwd_byts_b_avg*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙P

Identity_7Identity
Cast_5:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙X

Identity_8Identityinputs_bwd_iat_max*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Y

Identity_9Identityinputs_bwd_iat_mean*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
Identity_10Identityinputs_bwd_iat_min*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
Identity_11Identityinputs_bwd_iat_std*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
Identity_12Identityinputs_bwd_iat_tot*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙]
Identity_13Identityinputs_bwd_pkt_len_max*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙^
Identity_14Identityinputs_bwd_pkt_len_mean*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙]
Identity_15Identityinputs_bwd_pkt_len_min*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙]
Identity_16Identityinputs_bwd_pkt_len_std*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙\
Identity_17Identityinputs_bwd_pkts_b_avg*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙X
Identity_18Identityinputs_bwd_pkts_s*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
Identity_19Identity
Cast_7:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
Identity_20Identity
Cast_9:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙[
Identity_21Identityinputs_down_up_ratio*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
Identity_22Identity
Cast_1:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙R
Identity_23IdentityCast_16:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙R
Identity_24IdentityCast_10:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
Identity_25Identityinputs_flow_byts_s*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙[
Identity_26Identityinputs_flow_duration*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
Identity_27Identityinputs_flow_iat_max*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙[
Identity_28Identityinputs_flow_iat_mean*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
Identity_29Identityinputs_flow_iat_min*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
Identity_30Identityinputs_flow_iat_std*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
Identity_31Identityinputs_flow_pkts_s*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙_
Identity_32Identityinputs_fwd_act_data_pkts*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙^
Identity_33Identityinputs_fwd_blk_rate_avg*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙\
Identity_34Identityinputs_fwd_byts_b_avg*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
Identity_35Identity
Cast_4:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
Identity_36Identityinputs_fwd_iat_max*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
Identity_37Identityinputs_fwd_iat_mean*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
Identity_38Identityinputs_fwd_iat_min*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
Identity_39Identityinputs_fwd_iat_std*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
Identity_40Identityinputs_fwd_iat_tot*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙]
Identity_41Identityinputs_fwd_pkt_len_max*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙^
Identity_42Identityinputs_fwd_pkt_len_mean*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙]
Identity_43Identityinputs_fwd_pkt_len_min*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙]
Identity_44Identityinputs_fwd_pkt_len_std*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙\
Identity_45Identityinputs_fwd_pkts_b_avg*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙X
Identity_46Identityinputs_fwd_pkts_s*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
Identity_47Identity
Cast_6:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙^
Identity_48Identityinputs_fwd_seg_size_min*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
Identity_49Identity
Cast_8:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙V
Identity_50Identityinputs_idle_max*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙W
Identity_51Identityinputs_idle_mean*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙V
Identity_52Identityinputs_idle_min*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙V
Identity_53Identityinputs_idle_std*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙R
Identity_54Identityinputs_info*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙_
Identity_55Identityinputs_init_bwd_win_byts*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙_
Identity_56Identityinputs_init_fwd_win_byts*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
Identity_57Identityinputs_pkt_len_max*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
Identity_58Identityinputs_pkt_len_mean*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
Identity_59Identityinputs_pkt_len_min*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
Identity_60Identityinputs_pkt_len_std*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
Identity_61Identityinputs_pkt_len_var*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
Identity_62Identity
Cast_3:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
Identity_63Identityinputs_pkt_size_avg*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
Identity_64Identity
Cast_2:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙R
Identity_65IdentityCast_13:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙R
Identity_66IdentityCast_12:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙O
Identity_67IdentityCast:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙R
Identity_68IdentityCast_11:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
Identity_69Identityinputs_tot_bwd_pkts*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
Identity_70Identityinputs_tot_fwd_pkts*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙]
Identity_71Identityinputs_totlen_bwd_pkts*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙]
Identity_72Identityinputs_totlen_fwd_pkts*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙R
Identity_73IdentityCast_15:y:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0"#
identity_14Identity_14:output:0"#
identity_15Identity_15:output:0"#
identity_16Identity_16:output:0"#
identity_17Identity_17:output:0"#
identity_18Identity_18:output:0"#
identity_19Identity_19:output:0"!

identity_1Identity_1:output:0"#
identity_20Identity_20:output:0"#
identity_21Identity_21:output:0"#
identity_22Identity_22:output:0"#
identity_23Identity_23:output:0"#
identity_24Identity_24:output:0"#
identity_25Identity_25:output:0"#
identity_26Identity_26:output:0"#
identity_27Identity_27:output:0"#
identity_28Identity_28:output:0"#
identity_29Identity_29:output:0"!

identity_2Identity_2:output:0"#
identity_30Identity_30:output:0"#
identity_31Identity_31:output:0"#
identity_32Identity_32:output:0"#
identity_33Identity_33:output:0"#
identity_34Identity_34:output:0"#
identity_35Identity_35:output:0"#
identity_36Identity_36:output:0"#
identity_37Identity_37:output:0"#
identity_38Identity_38:output:0"#
identity_39Identity_39:output:0"!

identity_3Identity_3:output:0"#
identity_40Identity_40:output:0"#
identity_41Identity_41:output:0"#
identity_42Identity_42:output:0"#
identity_43Identity_43:output:0"#
identity_44Identity_44:output:0"#
identity_45Identity_45:output:0"#
identity_46Identity_46:output:0"#
identity_47Identity_47:output:0"#
identity_48Identity_48:output:0"#
identity_49Identity_49:output:0"!

identity_4Identity_4:output:0"#
identity_50Identity_50:output:0"#
identity_51Identity_51:output:0"#
identity_52Identity_52:output:0"#
identity_53Identity_53:output:0"#
identity_54Identity_54:output:0"#
identity_55Identity_55:output:0"#
identity_56Identity_56:output:0"#
identity_57Identity_57:output:0"#
identity_58Identity_58:output:0"#
identity_59Identity_59:output:0"!

identity_5Identity_5:output:0"#
identity_60Identity_60:output:0"#
identity_61Identity_61:output:0"#
identity_62Identity_62:output:0"#
identity_63Identity_63:output:0"#
identity_64Identity_64:output:0"#
identity_65Identity_65:output:0"#
identity_66Identity_66:output:0"#
identity_67Identity_67:output:0"#
identity_68Identity_68:output:0"#
identity_69Identity_69:output:0"!

identity_6Identity_6:output:0"#
identity_70Identity_70:output:0"#
identity_71Identity_71:output:0"#
identity_72Identity_72:output:0"#
identity_73Identity_73:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*§	
_input_shapes	
	:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:XMT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_urg_flag_cnt:[LW
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs_totlen_fwd_pkts:[KW
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs_totlen_bwd_pkts:XJT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_tot_fwd_pkts:XIT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_tot_bwd_pkts:UHQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameinputs_timestamp:XGT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_syn_flag_cnt:TFP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameinputs_src_port:REN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameinputs_src_ip:XDT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_rst_flag_cnt:XCT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_psh_flag_cnt:TBP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameinputs_protocol:XAT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_pkt_size_avg:V@R
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_pkt_length:W?S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_pkt_len_var:W>S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_pkt_len_std:W=S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_pkt_len_min:X<T
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_pkt_len_mean:W;S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_pkt_len_max:]:Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_nameinputs_init_fwd_win_byts:]9Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_nameinputs_init_bwd_win_byts:P8L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameinputs_info:T7P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameinputs_idle_std:T6P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameinputs_idle_min:U5Q
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameinputs_idle_mean:T4P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameinputs_idle_max:Y3U
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_fwd_urg_flags:\2X
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_nameinputs_fwd_seg_size_min:Y1U
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_fwd_psh_flags:V0R
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_fwd_pkts_s:Z/V
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs_fwd_pkts_b_avg:[.W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs_fwd_pkt_len_std:[-W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs_fwd_pkt_len_min:\,X
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_nameinputs_fwd_pkt_len_mean:[+W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs_fwd_pkt_len_max:W*S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_fwd_iat_tot:W)S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_fwd_iat_std:W(S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_fwd_iat_min:X'T
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_fwd_iat_mean:W&S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_fwd_iat_max:Z%V
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs_fwd_header_len:Z$V
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs_fwd_byts_b_avg:\#X
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_nameinputs_fwd_blk_rate_avg:]"Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_nameinputs_fwd_act_data_pkts:W!S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_flow_pkts_s:X T
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_flow_iat_std:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_flow_iat_min:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_flow_iat_mean:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_flow_iat_max:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_flow_duration:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_flow_byts_s:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_fin_flag_cnt:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_ece_flag_cnt:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameinputs_dst_port:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameinputs_dst_ip:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_down_up_ratio:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_bwd_urg_flags:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs_bwd_psh_flags:VR
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_bwd_pkts_s:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs_bwd_pkts_b_avg:[W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs_bwd_pkt_len_std:[W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs_bwd_pkt_len_min:\X
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_nameinputs_bwd_pkt_len_mean:[W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs_bwd_pkt_len_max:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_bwd_iat_tot:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_bwd_iat_std:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_bwd_iat_min:X
T
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_bwd_iat_mean:W	S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_bwd_iat_max:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs_bwd_header_len:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs_bwd_byts_b_avg:\X
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_nameinputs_bwd_blk_rate_avg:VR
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_active_std:VR
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_active_min:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs_active_mean:VR
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinputs_active_max:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs_ack_flag_cnt:N J
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	inputs_no
Ąw
˙
K__inference_RandomForestModel_layer_call_and_return_conditional_losses_3645
no
ack_flag_cnt

active_max
active_mean

active_min

active_std
bwd_blk_rate_avg
bwd_byts_b_avg
bwd_header_len
bwd_iat_max
bwd_iat_mean
bwd_iat_min
bwd_iat_std
bwd_iat_tot
bwd_pkt_len_max
bwd_pkt_len_mean
bwd_pkt_len_min
bwd_pkt_len_std
bwd_pkts_b_avg

bwd_pkts_s
bwd_psh_flags
bwd_urg_flags
down_up_ratio

dst_ip	
dst_port
ece_flag_cnt
fin_flag_cnt
flow_byts_s
flow_duration
flow_iat_max
flow_iat_mean
flow_iat_min
flow_iat_std
flow_pkts_s
fwd_act_data_pkts
fwd_blk_rate_avg
fwd_byts_b_avg
fwd_header_len
fwd_iat_max
fwd_iat_mean
fwd_iat_min
fwd_iat_std
fwd_iat_tot
fwd_pkt_len_max
fwd_pkt_len_mean
fwd_pkt_len_min
fwd_pkt_len_std
fwd_pkts_b_avg

fwd_pkts_s
fwd_psh_flags
fwd_seg_size_min
fwd_urg_flags
idle_max
	idle_mean
idle_min
idle_std
info
init_bwd_win_byts
init_fwd_win_byts
pkt_len_max
pkt_len_mean
pkt_len_min
pkt_len_std
pkt_len_var

pkt_length
pkt_size_avg
protocol
psh_flag_cnt
rst_flag_cnt

src_ip	
src_port
syn_flag_cnt
	timestamp
tot_bwd_pkts
tot_fwd_pkts
totlen_bwd_pkts
totlen_fwd_pkts
urg_flag_cnt
inference_op_model_handle
identity˘inference_opŚ
PartitionedCallPartitionedCallnoack_flag_cnt
active_maxactive_mean
active_min
active_stdbwd_blk_rate_avgbwd_byts_b_avgbwd_header_lenbwd_iat_maxbwd_iat_meanbwd_iat_minbwd_iat_stdbwd_iat_totbwd_pkt_len_maxbwd_pkt_len_meanbwd_pkt_len_minbwd_pkt_len_stdbwd_pkts_b_avg
bwd_pkts_sbwd_psh_flagsbwd_urg_flagsdown_up_ratiodst_ipdst_portece_flag_cntfin_flag_cntflow_byts_sflow_durationflow_iat_maxflow_iat_meanflow_iat_minflow_iat_stdflow_pkts_sfwd_act_data_pktsfwd_blk_rate_avgfwd_byts_b_avgfwd_header_lenfwd_iat_maxfwd_iat_meanfwd_iat_minfwd_iat_stdfwd_iat_totfwd_pkt_len_maxfwd_pkt_len_meanfwd_pkt_len_minfwd_pkt_len_stdfwd_pkts_b_avg
fwd_pkts_sfwd_psh_flagsfwd_seg_size_minfwd_urg_flagsidle_max	idle_meanidle_minidle_stdinfoinit_bwd_win_bytsinit_fwd_win_bytspkt_len_maxpkt_len_meanpkt_len_minpkt_len_stdpkt_len_var
pkt_lengthpkt_size_avgprotocolpsh_flag_cntrst_flag_cntsrc_ipsrc_portsyn_flag_cnt	timestamptot_bwd_pktstot_fwd_pktstotlen_bwd_pktstotlen_fwd_pktsurg_flag_cnt*Y
TinR
P2N		*V
ToutN
L2J*
_collective_manager_ids
 *ě
_output_shapesŮ
Ö:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *2
f-R+
)__inference__build_normalized_inputs_3215
stackPackPartitionedCall:output:0PartitionedCall:output:1PartitionedCall:output:2PartitionedCall:output:3PartitionedCall:output:4PartitionedCall:output:5PartitionedCall:output:6PartitionedCall:output:7PartitionedCall:output:8PartitionedCall:output:9PartitionedCall:output:10PartitionedCall:output:11PartitionedCall:output:12PartitionedCall:output:13PartitionedCall:output:14PartitionedCall:output:15PartitionedCall:output:16PartitionedCall:output:17PartitionedCall:output:18PartitionedCall:output:19PartitionedCall:output:20PartitionedCall:output:21PartitionedCall:output:22PartitionedCall:output:23PartitionedCall:output:24PartitionedCall:output:25PartitionedCall:output:26PartitionedCall:output:27PartitionedCall:output:28PartitionedCall:output:29PartitionedCall:output:30PartitionedCall:output:31PartitionedCall:output:32PartitionedCall:output:33PartitionedCall:output:34PartitionedCall:output:35PartitionedCall:output:36PartitionedCall:output:37PartitionedCall:output:38PartitionedCall:output:39PartitionedCall:output:40PartitionedCall:output:41PartitionedCall:output:42PartitionedCall:output:43PartitionedCall:output:44PartitionedCall:output:45PartitionedCall:output:46PartitionedCall:output:47PartitionedCall:output:48PartitionedCall:output:49PartitionedCall:output:50PartitionedCall:output:51PartitionedCall:output:52PartitionedCall:output:53PartitionedCall:output:54PartitionedCall:output:55PartitionedCall:output:56PartitionedCall:output:57PartitionedCall:output:58PartitionedCall:output:59PartitionedCall:output:60PartitionedCall:output:61PartitionedCall:output:62PartitionedCall:output:63PartitionedCall:output:64PartitionedCall:output:65PartitionedCall:output:66PartitionedCall:output:67PartitionedCall:output:68PartitionedCall:output:69PartitionedCall:output:70PartitionedCall:output:71PartitionedCall:output:72PartitionedCall:output:73*
NJ*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙J*

axisL
ConstConst*
_output_shapes
:  *
dtype0*
value
B  N
Const_1Const*
_output_shapes
:  *
dtype0*
value
B  X
RaggedConstant/valuesConst*
_output_shapes
: *
dtype0*
valueB ^
RaggedConstant/ConstConst*
_output_shapes
:*
dtype0	*
valueB	R `
RaggedConstant/Const_1Const*
_output_shapes
:*
dtype0	*
valueB	R Ą
inference_opSimpleMLInferenceOpWithHandlestack:output:0Const:output:0Const_1:output:0RaggedConstant/values:output:0RaggedConstant/Const:output:0RaggedConstant/Const_1:output:0inference_op_model_handle*-
_output_shapes
:˙˙˙˙˙˙˙˙˙:*
dense_output_dimÚ
PartitionedCall_1PartitionedCall inference_op:dense_predictions:0'inference_op:dense_col_representation:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 */
f*R(
&__inference__finalize_predictions_3307i
IdentityIdentityPartitionedCall_1:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙1
NoOpNoOp^inference_op*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Š	
_input_shapes	
	:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: 2
inference_opinference_op:,N(
&
_user_specified_namemodel_handle:QMM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameurg_flag_cnt:TLP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nametotlen_fwd_pkts:TKP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nametotlen_bwd_pkts:QJM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nametot_fwd_pkts:QIM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nametot_bwd_pkts:NHJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	timestamp:QGM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namesyn_flag_cnt:MFI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
src_port:KEG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namesrc_ip:QDM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namerst_flag_cnt:QCM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namepsh_flag_cnt:MBI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
protocol:QAM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namepkt_size_avg:O@K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
pkt_length:P?L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepkt_len_var:P>L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepkt_len_std:P=L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepkt_len_min:Q<M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namepkt_len_mean:P;L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepkt_len_max:V:R
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinit_fwd_win_byts:V9R
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinit_bwd_win_byts:I8E
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinfo:M7I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
idle_std:M6I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
idle_min:N5J
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	idle_mean:M4I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
idle_max:R3N
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namefwd_urg_flags:U2Q
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namefwd_seg_size_min:R1N
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namefwd_psh_flags:O0K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
fwd_pkts_s:S/O
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namefwd_pkts_b_avg:T.P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefwd_pkt_len_std:T-P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefwd_pkt_len_min:U,Q
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namefwd_pkt_len_mean:T+P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefwd_pkt_len_max:P*L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefwd_iat_tot:P)L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefwd_iat_std:P(L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefwd_iat_min:Q'M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefwd_iat_mean:P&L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefwd_iat_max:S%O
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namefwd_header_len:S$O
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namefwd_byts_b_avg:U#Q
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namefwd_blk_rate_avg:V"R
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_namefwd_act_data_pkts:P!L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameflow_pkts_s:Q M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameflow_iat_std:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameflow_iat_min:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameflow_iat_mean:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameflow_iat_max:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameflow_duration:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameflow_byts_s:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefin_flag_cnt:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameece_flag_cnt:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
dst_port:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namedst_ip:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namedown_up_ratio:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namebwd_urg_flags:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namebwd_psh_flags:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
bwd_pkts_s:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namebwd_pkts_b_avg:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namebwd_pkt_len_std:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namebwd_pkt_len_min:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namebwd_pkt_len_mean:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namebwd_pkt_len_max:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namebwd_iat_tot:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namebwd_iat_std:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namebwd_iat_min:Q
M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namebwd_iat_mean:P	L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namebwd_iat_max:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namebwd_header_len:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namebwd_byts_b_avg:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namebwd_blk_rate_avg:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
active_std:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
active_min:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameactive_mean:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
active_max:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameack_flag_cnt:G C
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNo
őU
Ý
0__inference_RandomForestModel_layer_call_fn_3813
no
ack_flag_cnt

active_max
active_mean

active_min

active_std
bwd_blk_rate_avg
bwd_byts_b_avg
bwd_header_len
bwd_iat_max
bwd_iat_mean
bwd_iat_min
bwd_iat_std
bwd_iat_tot
bwd_pkt_len_max
bwd_pkt_len_mean
bwd_pkt_len_min
bwd_pkt_len_std
bwd_pkts_b_avg

bwd_pkts_s
bwd_psh_flags
bwd_urg_flags
down_up_ratio

dst_ip	
dst_port
ece_flag_cnt
fin_flag_cnt
flow_byts_s
flow_duration
flow_iat_max
flow_iat_mean
flow_iat_min
flow_iat_std
flow_pkts_s
fwd_act_data_pkts
fwd_blk_rate_avg
fwd_byts_b_avg
fwd_header_len
fwd_iat_max
fwd_iat_mean
fwd_iat_min
fwd_iat_std
fwd_iat_tot
fwd_pkt_len_max
fwd_pkt_len_mean
fwd_pkt_len_min
fwd_pkt_len_std
fwd_pkts_b_avg

fwd_pkts_s
fwd_psh_flags
fwd_seg_size_min
fwd_urg_flags
idle_max
	idle_mean
idle_min
idle_std
info
init_bwd_win_byts
init_fwd_win_byts
pkt_len_max
pkt_len_mean
pkt_len_min
pkt_len_std
pkt_len_var

pkt_length
pkt_size_avg
protocol
psh_flag_cnt
rst_flag_cnt

src_ip	
src_port
syn_flag_cnt
	timestamp
tot_bwd_pkts
tot_fwd_pkts
totlen_bwd_pkts
totlen_fwd_pkts
urg_flag_cnt
unknown
identity˘StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallnoack_flag_cnt
active_maxactive_mean
active_min
active_stdbwd_blk_rate_avgbwd_byts_b_avgbwd_header_lenbwd_iat_maxbwd_iat_meanbwd_iat_minbwd_iat_stdbwd_iat_totbwd_pkt_len_maxbwd_pkt_len_meanbwd_pkt_len_minbwd_pkt_len_stdbwd_pkts_b_avg
bwd_pkts_sbwd_psh_flagsbwd_urg_flagsdown_up_ratiodst_ipdst_portece_flag_cntfin_flag_cntflow_byts_sflow_durationflow_iat_maxflow_iat_meanflow_iat_minflow_iat_stdflow_pkts_sfwd_act_data_pktsfwd_blk_rate_avgfwd_byts_b_avgfwd_header_lenfwd_iat_maxfwd_iat_meanfwd_iat_minfwd_iat_stdfwd_iat_totfwd_pkt_len_maxfwd_pkt_len_meanfwd_pkt_len_minfwd_pkt_len_stdfwd_pkts_b_avg
fwd_pkts_sfwd_psh_flagsfwd_seg_size_minfwd_urg_flagsidle_max	idle_meanidle_minidle_stdinfoinit_bwd_win_bytsinit_fwd_win_bytspkt_len_maxpkt_len_meanpkt_len_minpkt_len_stdpkt_len_var
pkt_lengthpkt_size_avgprotocolpsh_flag_cntrst_flag_cntsrc_ipsrc_portsyn_flag_cnt	timestamptot_bwd_pktstot_fwd_pktstotlen_bwd_pktstotlen_fwd_pktsurg_flag_cntunknown*Z
TinS
Q2O		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_RandomForestModel_layer_call_and_return_conditional_losses_3645o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Š	
_input_shapes	
	:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: 22
StatefulPartitionedCallStatefulPartitionedCall:$N 

_user_specified_name3809:QMM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameurg_flag_cnt:TLP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nametotlen_fwd_pkts:TKP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nametotlen_bwd_pkts:QJM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nametot_fwd_pkts:QIM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nametot_bwd_pkts:NHJ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	timestamp:QGM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namesyn_flag_cnt:MFI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
src_port:KEG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namesrc_ip:QDM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namerst_flag_cnt:QCM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namepsh_flag_cnt:MBI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
protocol:QAM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namepkt_size_avg:O@K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
pkt_length:P?L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepkt_len_var:P>L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepkt_len_std:P=L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepkt_len_min:Q<M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namepkt_len_mean:P;L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namepkt_len_max:V:R
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinit_fwd_win_byts:V9R
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_nameinit_bwd_win_byts:I8E
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameinfo:M7I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
idle_std:M6I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
idle_min:N5J
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	idle_mean:M4I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
idle_max:R3N
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namefwd_urg_flags:U2Q
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namefwd_seg_size_min:R1N
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namefwd_psh_flags:O0K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
fwd_pkts_s:S/O
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namefwd_pkts_b_avg:T.P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefwd_pkt_len_std:T-P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefwd_pkt_len_min:U,Q
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namefwd_pkt_len_mean:T+P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefwd_pkt_len_max:P*L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefwd_iat_tot:P)L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefwd_iat_std:P(L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefwd_iat_min:Q'M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefwd_iat_mean:P&L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefwd_iat_max:S%O
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namefwd_header_len:S$O
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namefwd_byts_b_avg:U#Q
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namefwd_blk_rate_avg:V"R
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_namefwd_act_data_pkts:P!L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameflow_pkts_s:Q M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameflow_iat_std:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameflow_iat_min:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameflow_iat_mean:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameflow_iat_max:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameflow_duration:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameflow_byts_s:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefin_flag_cnt:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameece_flag_cnt:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
dst_port:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namedst_ip:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namedown_up_ratio:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namebwd_urg_flags:RN
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_namebwd_psh_flags:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
bwd_pkts_s:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namebwd_pkts_b_avg:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namebwd_pkt_len_std:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namebwd_pkt_len_min:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namebwd_pkt_len_mean:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namebwd_pkt_len_max:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namebwd_iat_tot:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namebwd_iat_std:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namebwd_iat_min:Q
M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namebwd_iat_mean:P	L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namebwd_iat_max:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namebwd_header_len:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namebwd_byts_b_avg:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namebwd_blk_rate_avg:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
active_std:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
active_min:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameactive_mean:OK
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
active_max:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_nameack_flag_cnt:G C
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameNo
˝
Z
,__inference_yggdrasil_model_path_tensor_4165
staticregexreplace_input
identity
StaticRegexReplaceStaticRegexReplacestaticregexreplace_input*
_output_shapes
: *!
pattern7d524b07892540dadone*
rewrite R
IdentityIdentityStaticRegexReplace:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: "ĘL
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ć)
serving_default˛)
-
No'
serving_default_No:0˙˙˙˙˙˙˙˙˙
A
ack_flag_cnt1
serving_default_ack_flag_cnt:0˙˙˙˙˙˙˙˙˙
=

active_max/
serving_default_active_max:0˙˙˙˙˙˙˙˙˙
?
active_mean0
serving_default_active_mean:0˙˙˙˙˙˙˙˙˙
=

active_min/
serving_default_active_min:0˙˙˙˙˙˙˙˙˙
=

active_std/
serving_default_active_std:0˙˙˙˙˙˙˙˙˙
I
bwd_blk_rate_avg5
"serving_default_bwd_blk_rate_avg:0˙˙˙˙˙˙˙˙˙
E
bwd_byts_b_avg3
 serving_default_bwd_byts_b_avg:0˙˙˙˙˙˙˙˙˙
E
bwd_header_len3
 serving_default_bwd_header_len:0˙˙˙˙˙˙˙˙˙
?
bwd_iat_max0
serving_default_bwd_iat_max:0˙˙˙˙˙˙˙˙˙
A
bwd_iat_mean1
serving_default_bwd_iat_mean:0˙˙˙˙˙˙˙˙˙
?
bwd_iat_min0
serving_default_bwd_iat_min:0˙˙˙˙˙˙˙˙˙
?
bwd_iat_std0
serving_default_bwd_iat_std:0˙˙˙˙˙˙˙˙˙
?
bwd_iat_tot0
serving_default_bwd_iat_tot:0˙˙˙˙˙˙˙˙˙
G
bwd_pkt_len_max4
!serving_default_bwd_pkt_len_max:0˙˙˙˙˙˙˙˙˙
I
bwd_pkt_len_mean5
"serving_default_bwd_pkt_len_mean:0˙˙˙˙˙˙˙˙˙
G
bwd_pkt_len_min4
!serving_default_bwd_pkt_len_min:0˙˙˙˙˙˙˙˙˙
G
bwd_pkt_len_std4
!serving_default_bwd_pkt_len_std:0˙˙˙˙˙˙˙˙˙
E
bwd_pkts_b_avg3
 serving_default_bwd_pkts_b_avg:0˙˙˙˙˙˙˙˙˙
=

bwd_pkts_s/
serving_default_bwd_pkts_s:0˙˙˙˙˙˙˙˙˙
C
bwd_psh_flags2
serving_default_bwd_psh_flags:0˙˙˙˙˙˙˙˙˙
C
bwd_urg_flags2
serving_default_bwd_urg_flags:0˙˙˙˙˙˙˙˙˙
C
down_up_ratio2
serving_default_down_up_ratio:0˙˙˙˙˙˙˙˙˙
5
dst_ip+
serving_default_dst_ip:0	˙˙˙˙˙˙˙˙˙
9
dst_port-
serving_default_dst_port:0˙˙˙˙˙˙˙˙˙
A
ece_flag_cnt1
serving_default_ece_flag_cnt:0˙˙˙˙˙˙˙˙˙
A
fin_flag_cnt1
serving_default_fin_flag_cnt:0˙˙˙˙˙˙˙˙˙
?
flow_byts_s0
serving_default_flow_byts_s:0˙˙˙˙˙˙˙˙˙
C
flow_duration2
serving_default_flow_duration:0˙˙˙˙˙˙˙˙˙
A
flow_iat_max1
serving_default_flow_iat_max:0˙˙˙˙˙˙˙˙˙
C
flow_iat_mean2
serving_default_flow_iat_mean:0˙˙˙˙˙˙˙˙˙
A
flow_iat_min1
serving_default_flow_iat_min:0˙˙˙˙˙˙˙˙˙
A
flow_iat_std1
serving_default_flow_iat_std:0˙˙˙˙˙˙˙˙˙
?
flow_pkts_s0
serving_default_flow_pkts_s:0˙˙˙˙˙˙˙˙˙
K
fwd_act_data_pkts6
#serving_default_fwd_act_data_pkts:0˙˙˙˙˙˙˙˙˙
I
fwd_blk_rate_avg5
"serving_default_fwd_blk_rate_avg:0˙˙˙˙˙˙˙˙˙
E
fwd_byts_b_avg3
 serving_default_fwd_byts_b_avg:0˙˙˙˙˙˙˙˙˙
E
fwd_header_len3
 serving_default_fwd_header_len:0˙˙˙˙˙˙˙˙˙
?
fwd_iat_max0
serving_default_fwd_iat_max:0˙˙˙˙˙˙˙˙˙
A
fwd_iat_mean1
serving_default_fwd_iat_mean:0˙˙˙˙˙˙˙˙˙
?
fwd_iat_min0
serving_default_fwd_iat_min:0˙˙˙˙˙˙˙˙˙
?
fwd_iat_std0
serving_default_fwd_iat_std:0˙˙˙˙˙˙˙˙˙
?
fwd_iat_tot0
serving_default_fwd_iat_tot:0˙˙˙˙˙˙˙˙˙
G
fwd_pkt_len_max4
!serving_default_fwd_pkt_len_max:0˙˙˙˙˙˙˙˙˙
I
fwd_pkt_len_mean5
"serving_default_fwd_pkt_len_mean:0˙˙˙˙˙˙˙˙˙
G
fwd_pkt_len_min4
!serving_default_fwd_pkt_len_min:0˙˙˙˙˙˙˙˙˙
G
fwd_pkt_len_std4
!serving_default_fwd_pkt_len_std:0˙˙˙˙˙˙˙˙˙
E
fwd_pkts_b_avg3
 serving_default_fwd_pkts_b_avg:0˙˙˙˙˙˙˙˙˙
=

fwd_pkts_s/
serving_default_fwd_pkts_s:0˙˙˙˙˙˙˙˙˙
C
fwd_psh_flags2
serving_default_fwd_psh_flags:0˙˙˙˙˙˙˙˙˙
I
fwd_seg_size_min5
"serving_default_fwd_seg_size_min:0˙˙˙˙˙˙˙˙˙
C
fwd_urg_flags2
serving_default_fwd_urg_flags:0˙˙˙˙˙˙˙˙˙
9
idle_max-
serving_default_idle_max:0˙˙˙˙˙˙˙˙˙
;
	idle_mean.
serving_default_idle_mean:0˙˙˙˙˙˙˙˙˙
9
idle_min-
serving_default_idle_min:0˙˙˙˙˙˙˙˙˙
9
idle_std-
serving_default_idle_std:0˙˙˙˙˙˙˙˙˙
1
info)
serving_default_info:0˙˙˙˙˙˙˙˙˙
K
init_bwd_win_byts6
#serving_default_init_bwd_win_byts:0˙˙˙˙˙˙˙˙˙
K
init_fwd_win_byts6
#serving_default_init_fwd_win_byts:0˙˙˙˙˙˙˙˙˙
?
pkt_len_max0
serving_default_pkt_len_max:0˙˙˙˙˙˙˙˙˙
A
pkt_len_mean1
serving_default_pkt_len_mean:0˙˙˙˙˙˙˙˙˙
?
pkt_len_min0
serving_default_pkt_len_min:0˙˙˙˙˙˙˙˙˙
?
pkt_len_std0
serving_default_pkt_len_std:0˙˙˙˙˙˙˙˙˙
?
pkt_len_var0
serving_default_pkt_len_var:0˙˙˙˙˙˙˙˙˙
=

pkt_length/
serving_default_pkt_length:0˙˙˙˙˙˙˙˙˙
A
pkt_size_avg1
serving_default_pkt_size_avg:0˙˙˙˙˙˙˙˙˙
9
protocol-
serving_default_protocol:0˙˙˙˙˙˙˙˙˙
A
psh_flag_cnt1
serving_default_psh_flag_cnt:0˙˙˙˙˙˙˙˙˙
A
rst_flag_cnt1
serving_default_rst_flag_cnt:0˙˙˙˙˙˙˙˙˙
5
src_ip+
serving_default_src_ip:0	˙˙˙˙˙˙˙˙˙
9
src_port-
serving_default_src_port:0˙˙˙˙˙˙˙˙˙
A
syn_flag_cnt1
serving_default_syn_flag_cnt:0˙˙˙˙˙˙˙˙˙
;
	timestamp.
serving_default_timestamp:0˙˙˙˙˙˙˙˙˙
A
tot_bwd_pkts1
serving_default_tot_bwd_pkts:0˙˙˙˙˙˙˙˙˙
A
tot_fwd_pkts1
serving_default_tot_fwd_pkts:0˙˙˙˙˙˙˙˙˙
G
totlen_bwd_pkts4
!serving_default_totlen_bwd_pkts:0˙˙˙˙˙˙˙˙˙
G
totlen_fwd_pkts4
!serving_default_totlen_fwd_pkts:0˙˙˙˙˙˙˙˙˙
A
urg_flag_cnt1
serving_default_urg_flag_cnt:0˙˙˙˙˙˙˙˙˙<
output_10
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict22

asset_path_initializer:07d524b07892540dadone2<

asset_path_initializer_1:07d524b07892540dadata_spec.pb29

asset_path_initializer_2:07d524b07892540daheader.pb2D

asset_path_initializer_3:0$7d524b07892540danodes-00000-of-000012G

asset_path_initializer_4:0'7d524b07892540darandom_forest_header.pb:Ŕ
ś
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

_multitask
	_is_trained

_learner_params
	_features
	optimizer
loss
_models
_build_normalized_inputs
_finalize_predictions
call
call_get_leaves
yggdrasil_model_path_tensor

signatures"
_tf_keras_model
'
	0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ę
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ç
trace_0
trace_12
0__inference_RandomForestModel_layer_call_fn_3729
0__inference_RandomForestModel_layer_call_fn_3813Š
˘˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0ztrace_1
ý
trace_0
trace_12Ć
K__inference_RandomForestModel_layer_call_and_return_conditional_losses_3480
K__inference_RandomForestModel_layer_call_and_return_conditional_losses_3645Š
˘˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0ztrace_1
ü	Bů	
__inference__wrapped_model_3315Noack_flag_cnt
active_maxactive_mean
active_min
active_stdbwd_blk_rate_avgbwd_byts_b_avgbwd_header_lenbwd_iat_maxbwd_iat_meanbwd_iat_minbwd_iat_stdbwd_iat_totbwd_pkt_len_maxbwd_pkt_len_meanbwd_pkt_len_minbwd_pkt_len_stdbwd_pkts_b_avg
bwd_pkts_sbwd_psh_flagsbwd_urg_flagsdown_up_ratiodst_ipdst_portece_flag_cntfin_flag_cntflow_byts_sflow_durationflow_iat_maxflow_iat_meanflow_iat_minflow_iat_stdflow_pkts_sfwd_act_data_pktsfwd_blk_rate_avgfwd_byts_b_avgfwd_header_lenfwd_iat_maxfwd_iat_meanfwd_iat_minfwd_iat_stdfwd_iat_totfwd_pkt_len_maxfwd_pkt_len_meanfwd_pkt_len_minfwd_pkt_len_stdfwd_pkts_b_avg
fwd_pkts_sfwd_psh_flagsfwd_seg_size_minfwd_urg_flagsidle_max	idle_meanidle_minidle_stdinfoinit_bwd_win_bytsinit_fwd_win_bytspkt_len_maxpkt_len_meanpkt_len_minpkt_len_stdpkt_len_var
pkt_lengthpkt_size_avgprotocolpsh_flag_cntrst_flag_cntsrc_ipsrc_portsyn_flag_cnt	timestamptot_bwd_pktstot_fwd_pktstotlen_bwd_pktstotlen_fwd_pktsurg_flag_cntN"
˛
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
:
 2
is_trained
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
j

_variables
_iterations
 _learning_rate
!_update_step_xla"
experimentalOptimizer
 "
trackable_dict_wrapper
'
"0"
trackable_list_wrapper
ă
#trace_02Ć
)__inference__build_normalized_inputs_3986
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z#trace_0

$trace_02ä
&__inference__finalize_predictions_3995š
˛˛Ž
FullArgSpec1
args)&
jtask
jpredictions
jlike_engine
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z$trace_0
ŕ
%trace_02Ă
__inference_call_4160Š
˘˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z%trace_0
2
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ű
&trace_02Ţ
,__inference_yggdrasil_model_path_tensor_4165­
Ľ˛Ą
FullArgSpec$
args
jmultitask_model_index
varargs
 
varkw
 
defaults˘
` 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ z&trace_0
,
'serving_default"
signature_map
'
	0"
trackable_list_wrapper
 "
trackable_list_wrapper
<
(0
)1
*2
+3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper

B

0__inference_RandomForestModel_layer_call_fn_3729Noack_flag_cnt
active_maxactive_mean
active_min
active_stdbwd_blk_rate_avgbwd_byts_b_avgbwd_header_lenbwd_iat_maxbwd_iat_meanbwd_iat_minbwd_iat_stdbwd_iat_totbwd_pkt_len_maxbwd_pkt_len_meanbwd_pkt_len_minbwd_pkt_len_stdbwd_pkts_b_avg
bwd_pkts_sbwd_psh_flagsbwd_urg_flagsdown_up_ratiodst_ipdst_portece_flag_cntfin_flag_cntflow_byts_sflow_durationflow_iat_maxflow_iat_meanflow_iat_minflow_iat_stdflow_pkts_sfwd_act_data_pktsfwd_blk_rate_avgfwd_byts_b_avgfwd_header_lenfwd_iat_maxfwd_iat_meanfwd_iat_minfwd_iat_stdfwd_iat_totfwd_pkt_len_maxfwd_pkt_len_meanfwd_pkt_len_minfwd_pkt_len_stdfwd_pkts_b_avg
fwd_pkts_sfwd_psh_flagsfwd_seg_size_minfwd_urg_flagsidle_max	idle_meanidle_minidle_stdinfoinit_bwd_win_bytsinit_fwd_win_bytspkt_len_maxpkt_len_meanpkt_len_minpkt_len_stdpkt_len_var
pkt_lengthpkt_size_avgprotocolpsh_flag_cntrst_flag_cntsrc_ipsrc_portsyn_flag_cnt	timestamptot_bwd_pktstot_fwd_pktstotlen_bwd_pktstotlen_fwd_pktsurg_flag_cntN"Š
˘˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 

B

0__inference_RandomForestModel_layer_call_fn_3813Noack_flag_cnt
active_maxactive_mean
active_min
active_stdbwd_blk_rate_avgbwd_byts_b_avgbwd_header_lenbwd_iat_maxbwd_iat_meanbwd_iat_minbwd_iat_stdbwd_iat_totbwd_pkt_len_maxbwd_pkt_len_meanbwd_pkt_len_minbwd_pkt_len_stdbwd_pkts_b_avg
bwd_pkts_sbwd_psh_flagsbwd_urg_flagsdown_up_ratiodst_ipdst_portece_flag_cntfin_flag_cntflow_byts_sflow_durationflow_iat_maxflow_iat_meanflow_iat_minflow_iat_stdflow_pkts_sfwd_act_data_pktsfwd_blk_rate_avgfwd_byts_b_avgfwd_header_lenfwd_iat_maxfwd_iat_meanfwd_iat_minfwd_iat_stdfwd_iat_totfwd_pkt_len_maxfwd_pkt_len_meanfwd_pkt_len_minfwd_pkt_len_stdfwd_pkts_b_avg
fwd_pkts_sfwd_psh_flagsfwd_seg_size_minfwd_urg_flagsidle_max	idle_meanidle_minidle_stdinfoinit_bwd_win_bytsinit_fwd_win_bytspkt_len_maxpkt_len_meanpkt_len_minpkt_len_stdpkt_len_var
pkt_lengthpkt_size_avgprotocolpsh_flag_cntrst_flag_cntsrc_ipsrc_portsyn_flag_cnt	timestamptot_bwd_pktstot_fwd_pktstotlen_bwd_pktstotlen_fwd_pktsurg_flag_cntN"Š
˘˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
š
Bś

K__inference_RandomForestModel_layer_call_and_return_conditional_losses_3480Noack_flag_cnt
active_maxactive_mean
active_min
active_stdbwd_blk_rate_avgbwd_byts_b_avgbwd_header_lenbwd_iat_maxbwd_iat_meanbwd_iat_minbwd_iat_stdbwd_iat_totbwd_pkt_len_maxbwd_pkt_len_meanbwd_pkt_len_minbwd_pkt_len_stdbwd_pkts_b_avg
bwd_pkts_sbwd_psh_flagsbwd_urg_flagsdown_up_ratiodst_ipdst_portece_flag_cntfin_flag_cntflow_byts_sflow_durationflow_iat_maxflow_iat_meanflow_iat_minflow_iat_stdflow_pkts_sfwd_act_data_pktsfwd_blk_rate_avgfwd_byts_b_avgfwd_header_lenfwd_iat_maxfwd_iat_meanfwd_iat_minfwd_iat_stdfwd_iat_totfwd_pkt_len_maxfwd_pkt_len_meanfwd_pkt_len_minfwd_pkt_len_stdfwd_pkts_b_avg
fwd_pkts_sfwd_psh_flagsfwd_seg_size_minfwd_urg_flagsidle_max	idle_meanidle_minidle_stdinfoinit_bwd_win_bytsinit_fwd_win_bytspkt_len_maxpkt_len_meanpkt_len_minpkt_len_stdpkt_len_var
pkt_lengthpkt_size_avgprotocolpsh_flag_cntrst_flag_cntsrc_ipsrc_portsyn_flag_cnt	timestamptot_bwd_pktstot_fwd_pktstotlen_bwd_pktstotlen_fwd_pktsurg_flag_cntN"Š
˘˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
š
Bś

K__inference_RandomForestModel_layer_call_and_return_conditional_losses_3645Noack_flag_cnt
active_maxactive_mean
active_min
active_stdbwd_blk_rate_avgbwd_byts_b_avgbwd_header_lenbwd_iat_maxbwd_iat_meanbwd_iat_minbwd_iat_stdbwd_iat_totbwd_pkt_len_maxbwd_pkt_len_meanbwd_pkt_len_minbwd_pkt_len_stdbwd_pkts_b_avg
bwd_pkts_sbwd_psh_flagsbwd_urg_flagsdown_up_ratiodst_ipdst_portece_flag_cntfin_flag_cntflow_byts_sflow_durationflow_iat_maxflow_iat_meanflow_iat_minflow_iat_stdflow_pkts_sfwd_act_data_pktsfwd_blk_rate_avgfwd_byts_b_avgfwd_header_lenfwd_iat_maxfwd_iat_meanfwd_iat_minfwd_iat_stdfwd_iat_totfwd_pkt_len_maxfwd_pkt_len_meanfwd_pkt_len_minfwd_pkt_len_stdfwd_pkts_b_avg
fwd_pkts_sfwd_psh_flagsfwd_seg_size_minfwd_urg_flagsidle_max	idle_meanidle_minidle_stdinfoinit_bwd_win_bytsinit_fwd_win_bytspkt_len_maxpkt_len_meanpkt_len_minpkt_len_stdpkt_len_var
pkt_lengthpkt_size_avgprotocolpsh_flag_cntrst_flag_cntsrc_ipsrc_portsyn_flag_cnt	timestamptot_bwd_pktstot_fwd_pktstotlen_bwd_pktstotlen_fwd_pktsurg_flag_cntN"Š
˘˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
'
0"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
ľ2˛Ż
Ś˛˘
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 0
G
,_input_builder
-_compiled_model"
_generic_user_object
¨BĽ
)__inference__build_normalized_inputs_3986	inputs_noinputs_ack_flag_cntinputs_active_maxinputs_active_meaninputs_active_mininputs_active_stdinputs_bwd_blk_rate_avginputs_bwd_byts_b_avginputs_bwd_header_leninputs_bwd_iat_maxinputs_bwd_iat_meaninputs_bwd_iat_mininputs_bwd_iat_stdinputs_bwd_iat_totinputs_bwd_pkt_len_maxinputs_bwd_pkt_len_meaninputs_bwd_pkt_len_mininputs_bwd_pkt_len_stdinputs_bwd_pkts_b_avginputs_bwd_pkts_sinputs_bwd_psh_flagsinputs_bwd_urg_flagsinputs_down_up_ratioinputs_dst_ipinputs_dst_portinputs_ece_flag_cntinputs_fin_flag_cntinputs_flow_byts_sinputs_flow_durationinputs_flow_iat_maxinputs_flow_iat_meaninputs_flow_iat_mininputs_flow_iat_stdinputs_flow_pkts_sinputs_fwd_act_data_pktsinputs_fwd_blk_rate_avginputs_fwd_byts_b_avginputs_fwd_header_leninputs_fwd_iat_maxinputs_fwd_iat_meaninputs_fwd_iat_mininputs_fwd_iat_stdinputs_fwd_iat_totinputs_fwd_pkt_len_maxinputs_fwd_pkt_len_meaninputs_fwd_pkt_len_mininputs_fwd_pkt_len_stdinputs_fwd_pkts_b_avginputs_fwd_pkts_sinputs_fwd_psh_flagsinputs_fwd_seg_size_mininputs_fwd_urg_flagsinputs_idle_maxinputs_idle_meaninputs_idle_mininputs_idle_stdinputs_infoinputs_init_bwd_win_bytsinputs_init_fwd_win_bytsinputs_pkt_len_maxinputs_pkt_len_meaninputs_pkt_len_mininputs_pkt_len_stdinputs_pkt_len_varinputs_pkt_lengthinputs_pkt_size_avginputs_protocolinputs_psh_flag_cntinputs_rst_flag_cntinputs_src_ipinputs_src_portinputs_syn_flag_cntinputs_timestampinputs_tot_bwd_pktsinputs_tot_fwd_pktsinputs_totlen_bwd_pktsinputs_totlen_fwd_pktsinputs_urg_flag_cntN"
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ŽBŤ
&__inference__finalize_predictions_3995predictions_dense_predictions$predictions_dense_col_representation"š
˛˛Ž
FullArgSpec1
args)&
jtask
jpredictions
jlike_engine
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ĽB˘
__inference_call_4160	inputs_noinputs_ack_flag_cntinputs_active_maxinputs_active_meaninputs_active_mininputs_active_stdinputs_bwd_blk_rate_avginputs_bwd_byts_b_avginputs_bwd_header_leninputs_bwd_iat_maxinputs_bwd_iat_meaninputs_bwd_iat_mininputs_bwd_iat_stdinputs_bwd_iat_totinputs_bwd_pkt_len_maxinputs_bwd_pkt_len_meaninputs_bwd_pkt_len_mininputs_bwd_pkt_len_stdinputs_bwd_pkts_b_avginputs_bwd_pkts_sinputs_bwd_psh_flagsinputs_bwd_urg_flagsinputs_down_up_ratioinputs_dst_ipinputs_dst_portinputs_ece_flag_cntinputs_fin_flag_cntinputs_flow_byts_sinputs_flow_durationinputs_flow_iat_maxinputs_flow_iat_meaninputs_flow_iat_mininputs_flow_iat_stdinputs_flow_pkts_sinputs_fwd_act_data_pktsinputs_fwd_blk_rate_avginputs_fwd_byts_b_avginputs_fwd_header_leninputs_fwd_iat_maxinputs_fwd_iat_meaninputs_fwd_iat_mininputs_fwd_iat_stdinputs_fwd_iat_totinputs_fwd_pkt_len_maxinputs_fwd_pkt_len_meaninputs_fwd_pkt_len_mininputs_fwd_pkt_len_stdinputs_fwd_pkts_b_avginputs_fwd_pkts_sinputs_fwd_psh_flagsinputs_fwd_seg_size_mininputs_fwd_urg_flagsinputs_idle_maxinputs_idle_meaninputs_idle_mininputs_idle_stdinputs_infoinputs_init_bwd_win_bytsinputs_init_fwd_win_bytsinputs_pkt_len_maxinputs_pkt_len_meaninputs_pkt_len_mininputs_pkt_len_stdinputs_pkt_len_varinputs_pkt_lengthinputs_pkt_size_avginputs_protocolinputs_psh_flag_cntinputs_rst_flag_cntinputs_src_ipinputs_src_portinputs_syn_flag_cntinputs_timestampinputs_tot_bwd_pktsinputs_tot_fwd_pktsinputs_totlen_bwd_pktsinputs_totlen_fwd_pktsinputs_urg_flag_cntN"Š
˘˛
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
˙
.	capture_0BŢ
,__inference_yggdrasil_model_path_tensor_4165"­
Ľ˛Ą
FullArgSpec$
args
jmultitask_model_index
varargs
 
varkw
 
defaults˘
` 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ z.	capture_0
ů	Bö	
"__inference_signature_wrapper_4250Noack_flag_cnt
active_maxactive_mean
active_min
active_stdbwd_blk_rate_avgbwd_byts_b_avgbwd_header_lenbwd_iat_maxbwd_iat_meanbwd_iat_minbwd_iat_stdbwd_iat_totbwd_pkt_len_maxbwd_pkt_len_meanbwd_pkt_len_minbwd_pkt_len_stdbwd_pkts_b_avg
bwd_pkts_sbwd_psh_flagsbwd_urg_flagsdown_up_ratiodst_ipdst_portece_flag_cntfin_flag_cntflow_byts_sflow_durationflow_iat_maxflow_iat_meanflow_iat_minflow_iat_stdflow_pkts_sfwd_act_data_pktsfwd_blk_rate_avgfwd_byts_b_avgfwd_header_lenfwd_iat_maxfwd_iat_meanfwd_iat_minfwd_iat_stdfwd_iat_totfwd_pkt_len_maxfwd_pkt_len_meanfwd_pkt_len_minfwd_pkt_len_stdfwd_pkts_b_avg
fwd_pkts_sfwd_psh_flagsfwd_seg_size_minfwd_urg_flagsidle_max	idle_meanidle_minidle_stdinfoinit_bwd_win_bytsinit_fwd_win_bytspkt_len_maxpkt_len_meanpkt_len_minpkt_len_stdpkt_len_var
pkt_lengthpkt_size_avgprotocolpsh_flag_cntrst_flag_cntsrc_ipsrc_portsyn_flag_cnt	timestamptot_bwd_pktstot_fwd_pktstotlen_bwd_pktstotlen_fwd_pktsurg_flag_cnt"
˛
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
N
/	variables
0	keras_api
	1total
	2count"
_tf_keras_metric
^
3	variables
4	keras_api
	5total
	6count
7
_fn_kwargs"
_tf_keras_metric
q
8	variables
9	keras_api
:
thresholds
;true_positives
<false_positives"
_tf_keras_metric
q
=	variables
>	keras_api
?
thresholds
@true_positives
Afalse_negatives"
_tf_keras_metric
l
B_feature_name_to_idx
C	_init_ops
#Dcategorical_str_to_int_hashmaps"
_generic_user_object
S
E_model_loader
F_create_resource
G_initialize
H_destroy_resourceR 
* 
.
10
21"
trackable_list_wrapper
-
/	variables"
_generic_user_object
:  (2total
:  (2count
.
50
61"
trackable_list_wrapper
-
3	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
;0
<1"
trackable_list_wrapper
-
8	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
.
@0
A1"
trackable_list_wrapper
-
=	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Q
I_output_types
J
_all_files
.
_done_file"
_generic_user_object
Ę
Ktrace_02­
__inference__creator_4254
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ zKtrace_0
Î
Ltrace_02ą
__inference__initializer_4261
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ zLtrace_0
Ě
Mtrace_02Ż
__inference__destroyer_4265
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ zMtrace_0
 "
trackable_list_wrapper
C
N0
O1
.2
P3
Q4"
trackable_list_wrapper
°B­
__inference__creator_4254"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
Ň
.	capture_0Bą
__inference__initializer_4261"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ z.	capture_0
˛BŻ
__inference__destroyer_4265"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
*
*
*
*ĺ 
K__inference_RandomForestModel_layer_call_and_return_conditional_losses_3480 -á˘Ý
Ő˘Ń
ĘŞĆ

No
No˙˙˙˙˙˙˙˙˙
2
ack_flag_cnt"
ack_flag_cnt˙˙˙˙˙˙˙˙˙
.

active_max 

active_max˙˙˙˙˙˙˙˙˙
0
active_mean!
active_mean˙˙˙˙˙˙˙˙˙
.

active_min 

active_min˙˙˙˙˙˙˙˙˙
.

active_std 

active_std˙˙˙˙˙˙˙˙˙
:
bwd_blk_rate_avg&#
bwd_blk_rate_avg˙˙˙˙˙˙˙˙˙
6
bwd_byts_b_avg$!
bwd_byts_b_avg˙˙˙˙˙˙˙˙˙
6
bwd_header_len$!
bwd_header_len˙˙˙˙˙˙˙˙˙
0
bwd_iat_max!
bwd_iat_max˙˙˙˙˙˙˙˙˙
2
bwd_iat_mean"
bwd_iat_mean˙˙˙˙˙˙˙˙˙
0
bwd_iat_min!
bwd_iat_min˙˙˙˙˙˙˙˙˙
0
bwd_iat_std!
bwd_iat_std˙˙˙˙˙˙˙˙˙
0
bwd_iat_tot!
bwd_iat_tot˙˙˙˙˙˙˙˙˙
8
bwd_pkt_len_max%"
bwd_pkt_len_max˙˙˙˙˙˙˙˙˙
:
bwd_pkt_len_mean&#
bwd_pkt_len_mean˙˙˙˙˙˙˙˙˙
8
bwd_pkt_len_min%"
bwd_pkt_len_min˙˙˙˙˙˙˙˙˙
8
bwd_pkt_len_std%"
bwd_pkt_len_std˙˙˙˙˙˙˙˙˙
6
bwd_pkts_b_avg$!
bwd_pkts_b_avg˙˙˙˙˙˙˙˙˙
.

bwd_pkts_s 

bwd_pkts_s˙˙˙˙˙˙˙˙˙
4
bwd_psh_flags# 
bwd_psh_flags˙˙˙˙˙˙˙˙˙
4
bwd_urg_flags# 
bwd_urg_flags˙˙˙˙˙˙˙˙˙
4
down_up_ratio# 
down_up_ratio˙˙˙˙˙˙˙˙˙
&
dst_ip
dst_ip˙˙˙˙˙˙˙˙˙	
*
dst_port
dst_port˙˙˙˙˙˙˙˙˙
2
ece_flag_cnt"
ece_flag_cnt˙˙˙˙˙˙˙˙˙
2
fin_flag_cnt"
fin_flag_cnt˙˙˙˙˙˙˙˙˙
0
flow_byts_s!
flow_byts_s˙˙˙˙˙˙˙˙˙
4
flow_duration# 
flow_duration˙˙˙˙˙˙˙˙˙
2
flow_iat_max"
flow_iat_max˙˙˙˙˙˙˙˙˙
4
flow_iat_mean# 
flow_iat_mean˙˙˙˙˙˙˙˙˙
2
flow_iat_min"
flow_iat_min˙˙˙˙˙˙˙˙˙
2
flow_iat_std"
flow_iat_std˙˙˙˙˙˙˙˙˙
0
flow_pkts_s!
flow_pkts_s˙˙˙˙˙˙˙˙˙
<
fwd_act_data_pkts'$
fwd_act_data_pkts˙˙˙˙˙˙˙˙˙
:
fwd_blk_rate_avg&#
fwd_blk_rate_avg˙˙˙˙˙˙˙˙˙
6
fwd_byts_b_avg$!
fwd_byts_b_avg˙˙˙˙˙˙˙˙˙
6
fwd_header_len$!
fwd_header_len˙˙˙˙˙˙˙˙˙
0
fwd_iat_max!
fwd_iat_max˙˙˙˙˙˙˙˙˙
2
fwd_iat_mean"
fwd_iat_mean˙˙˙˙˙˙˙˙˙
0
fwd_iat_min!
fwd_iat_min˙˙˙˙˙˙˙˙˙
0
fwd_iat_std!
fwd_iat_std˙˙˙˙˙˙˙˙˙
0
fwd_iat_tot!
fwd_iat_tot˙˙˙˙˙˙˙˙˙
8
fwd_pkt_len_max%"
fwd_pkt_len_max˙˙˙˙˙˙˙˙˙
:
fwd_pkt_len_mean&#
fwd_pkt_len_mean˙˙˙˙˙˙˙˙˙
8
fwd_pkt_len_min%"
fwd_pkt_len_min˙˙˙˙˙˙˙˙˙
8
fwd_pkt_len_std%"
fwd_pkt_len_std˙˙˙˙˙˙˙˙˙
6
fwd_pkts_b_avg$!
fwd_pkts_b_avg˙˙˙˙˙˙˙˙˙
.

fwd_pkts_s 

fwd_pkts_s˙˙˙˙˙˙˙˙˙
4
fwd_psh_flags# 
fwd_psh_flags˙˙˙˙˙˙˙˙˙
:
fwd_seg_size_min&#
fwd_seg_size_min˙˙˙˙˙˙˙˙˙
4
fwd_urg_flags# 
fwd_urg_flags˙˙˙˙˙˙˙˙˙
*
idle_max
idle_max˙˙˙˙˙˙˙˙˙
,
	idle_mean
	idle_mean˙˙˙˙˙˙˙˙˙
*
idle_min
idle_min˙˙˙˙˙˙˙˙˙
*
idle_std
idle_std˙˙˙˙˙˙˙˙˙
"
info
info˙˙˙˙˙˙˙˙˙
<
init_bwd_win_byts'$
init_bwd_win_byts˙˙˙˙˙˙˙˙˙
<
init_fwd_win_byts'$
init_fwd_win_byts˙˙˙˙˙˙˙˙˙
0
pkt_len_max!
pkt_len_max˙˙˙˙˙˙˙˙˙
2
pkt_len_mean"
pkt_len_mean˙˙˙˙˙˙˙˙˙
0
pkt_len_min!
pkt_len_min˙˙˙˙˙˙˙˙˙
0
pkt_len_std!
pkt_len_std˙˙˙˙˙˙˙˙˙
0
pkt_len_var!
pkt_len_var˙˙˙˙˙˙˙˙˙
.

pkt_length 

pkt_length˙˙˙˙˙˙˙˙˙
2
pkt_size_avg"
pkt_size_avg˙˙˙˙˙˙˙˙˙
*
protocol
protocol˙˙˙˙˙˙˙˙˙
2
psh_flag_cnt"
psh_flag_cnt˙˙˙˙˙˙˙˙˙
2
rst_flag_cnt"
rst_flag_cnt˙˙˙˙˙˙˙˙˙
&
src_ip
src_ip˙˙˙˙˙˙˙˙˙	
*
src_port
src_port˙˙˙˙˙˙˙˙˙
2
syn_flag_cnt"
syn_flag_cnt˙˙˙˙˙˙˙˙˙
,
	timestamp
	timestamp˙˙˙˙˙˙˙˙˙
2
tot_bwd_pkts"
tot_bwd_pkts˙˙˙˙˙˙˙˙˙
2
tot_fwd_pkts"
tot_fwd_pkts˙˙˙˙˙˙˙˙˙
8
totlen_bwd_pkts%"
totlen_bwd_pkts˙˙˙˙˙˙˙˙˙
8
totlen_fwd_pkts%"
totlen_fwd_pkts˙˙˙˙˙˙˙˙˙
2
urg_flag_cnt"
urg_flag_cnt˙˙˙˙˙˙˙˙˙
p
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 ĺ 
K__inference_RandomForestModel_layer_call_and_return_conditional_losses_3645 -á˘Ý
Ő˘Ń
ĘŞĆ

No
No˙˙˙˙˙˙˙˙˙
2
ack_flag_cnt"
ack_flag_cnt˙˙˙˙˙˙˙˙˙
.

active_max 

active_max˙˙˙˙˙˙˙˙˙
0
active_mean!
active_mean˙˙˙˙˙˙˙˙˙
.

active_min 

active_min˙˙˙˙˙˙˙˙˙
.

active_std 

active_std˙˙˙˙˙˙˙˙˙
:
bwd_blk_rate_avg&#
bwd_blk_rate_avg˙˙˙˙˙˙˙˙˙
6
bwd_byts_b_avg$!
bwd_byts_b_avg˙˙˙˙˙˙˙˙˙
6
bwd_header_len$!
bwd_header_len˙˙˙˙˙˙˙˙˙
0
bwd_iat_max!
bwd_iat_max˙˙˙˙˙˙˙˙˙
2
bwd_iat_mean"
bwd_iat_mean˙˙˙˙˙˙˙˙˙
0
bwd_iat_min!
bwd_iat_min˙˙˙˙˙˙˙˙˙
0
bwd_iat_std!
bwd_iat_std˙˙˙˙˙˙˙˙˙
0
bwd_iat_tot!
bwd_iat_tot˙˙˙˙˙˙˙˙˙
8
bwd_pkt_len_max%"
bwd_pkt_len_max˙˙˙˙˙˙˙˙˙
:
bwd_pkt_len_mean&#
bwd_pkt_len_mean˙˙˙˙˙˙˙˙˙
8
bwd_pkt_len_min%"
bwd_pkt_len_min˙˙˙˙˙˙˙˙˙
8
bwd_pkt_len_std%"
bwd_pkt_len_std˙˙˙˙˙˙˙˙˙
6
bwd_pkts_b_avg$!
bwd_pkts_b_avg˙˙˙˙˙˙˙˙˙
.

bwd_pkts_s 

bwd_pkts_s˙˙˙˙˙˙˙˙˙
4
bwd_psh_flags# 
bwd_psh_flags˙˙˙˙˙˙˙˙˙
4
bwd_urg_flags# 
bwd_urg_flags˙˙˙˙˙˙˙˙˙
4
down_up_ratio# 
down_up_ratio˙˙˙˙˙˙˙˙˙
&
dst_ip
dst_ip˙˙˙˙˙˙˙˙˙	
*
dst_port
dst_port˙˙˙˙˙˙˙˙˙
2
ece_flag_cnt"
ece_flag_cnt˙˙˙˙˙˙˙˙˙
2
fin_flag_cnt"
fin_flag_cnt˙˙˙˙˙˙˙˙˙
0
flow_byts_s!
flow_byts_s˙˙˙˙˙˙˙˙˙
4
flow_duration# 
flow_duration˙˙˙˙˙˙˙˙˙
2
flow_iat_max"
flow_iat_max˙˙˙˙˙˙˙˙˙
4
flow_iat_mean# 
flow_iat_mean˙˙˙˙˙˙˙˙˙
2
flow_iat_min"
flow_iat_min˙˙˙˙˙˙˙˙˙
2
flow_iat_std"
flow_iat_std˙˙˙˙˙˙˙˙˙
0
flow_pkts_s!
flow_pkts_s˙˙˙˙˙˙˙˙˙
<
fwd_act_data_pkts'$
fwd_act_data_pkts˙˙˙˙˙˙˙˙˙
:
fwd_blk_rate_avg&#
fwd_blk_rate_avg˙˙˙˙˙˙˙˙˙
6
fwd_byts_b_avg$!
fwd_byts_b_avg˙˙˙˙˙˙˙˙˙
6
fwd_header_len$!
fwd_header_len˙˙˙˙˙˙˙˙˙
0
fwd_iat_max!
fwd_iat_max˙˙˙˙˙˙˙˙˙
2
fwd_iat_mean"
fwd_iat_mean˙˙˙˙˙˙˙˙˙
0
fwd_iat_min!
fwd_iat_min˙˙˙˙˙˙˙˙˙
0
fwd_iat_std!
fwd_iat_std˙˙˙˙˙˙˙˙˙
0
fwd_iat_tot!
fwd_iat_tot˙˙˙˙˙˙˙˙˙
8
fwd_pkt_len_max%"
fwd_pkt_len_max˙˙˙˙˙˙˙˙˙
:
fwd_pkt_len_mean&#
fwd_pkt_len_mean˙˙˙˙˙˙˙˙˙
8
fwd_pkt_len_min%"
fwd_pkt_len_min˙˙˙˙˙˙˙˙˙
8
fwd_pkt_len_std%"
fwd_pkt_len_std˙˙˙˙˙˙˙˙˙
6
fwd_pkts_b_avg$!
fwd_pkts_b_avg˙˙˙˙˙˙˙˙˙
.

fwd_pkts_s 

fwd_pkts_s˙˙˙˙˙˙˙˙˙
4
fwd_psh_flags# 
fwd_psh_flags˙˙˙˙˙˙˙˙˙
:
fwd_seg_size_min&#
fwd_seg_size_min˙˙˙˙˙˙˙˙˙
4
fwd_urg_flags# 
fwd_urg_flags˙˙˙˙˙˙˙˙˙
*
idle_max
idle_max˙˙˙˙˙˙˙˙˙
,
	idle_mean
	idle_mean˙˙˙˙˙˙˙˙˙
*
idle_min
idle_min˙˙˙˙˙˙˙˙˙
*
idle_std
idle_std˙˙˙˙˙˙˙˙˙
"
info
info˙˙˙˙˙˙˙˙˙
<
init_bwd_win_byts'$
init_bwd_win_byts˙˙˙˙˙˙˙˙˙
<
init_fwd_win_byts'$
init_fwd_win_byts˙˙˙˙˙˙˙˙˙
0
pkt_len_max!
pkt_len_max˙˙˙˙˙˙˙˙˙
2
pkt_len_mean"
pkt_len_mean˙˙˙˙˙˙˙˙˙
0
pkt_len_min!
pkt_len_min˙˙˙˙˙˙˙˙˙
0
pkt_len_std!
pkt_len_std˙˙˙˙˙˙˙˙˙
0
pkt_len_var!
pkt_len_var˙˙˙˙˙˙˙˙˙
.

pkt_length 

pkt_length˙˙˙˙˙˙˙˙˙
2
pkt_size_avg"
pkt_size_avg˙˙˙˙˙˙˙˙˙
*
protocol
protocol˙˙˙˙˙˙˙˙˙
2
psh_flag_cnt"
psh_flag_cnt˙˙˙˙˙˙˙˙˙
2
rst_flag_cnt"
rst_flag_cnt˙˙˙˙˙˙˙˙˙
&
src_ip
src_ip˙˙˙˙˙˙˙˙˙	
*
src_port
src_port˙˙˙˙˙˙˙˙˙
2
syn_flag_cnt"
syn_flag_cnt˙˙˙˙˙˙˙˙˙
,
	timestamp
	timestamp˙˙˙˙˙˙˙˙˙
2
tot_bwd_pkts"
tot_bwd_pkts˙˙˙˙˙˙˙˙˙
2
tot_fwd_pkts"
tot_fwd_pkts˙˙˙˙˙˙˙˙˙
8
totlen_bwd_pkts%"
totlen_bwd_pkts˙˙˙˙˙˙˙˙˙
8
totlen_fwd_pkts%"
totlen_fwd_pkts˙˙˙˙˙˙˙˙˙
2
urg_flag_cnt"
urg_flag_cnt˙˙˙˙˙˙˙˙˙
p 
Ş ",˘)
"
tensor_0˙˙˙˙˙˙˙˙˙
 ż 
0__inference_RandomForestModel_layer_call_fn_3729 -á˘Ý
Ő˘Ń
ĘŞĆ

No
No˙˙˙˙˙˙˙˙˙
2
ack_flag_cnt"
ack_flag_cnt˙˙˙˙˙˙˙˙˙
.

active_max 

active_max˙˙˙˙˙˙˙˙˙
0
active_mean!
active_mean˙˙˙˙˙˙˙˙˙
.

active_min 

active_min˙˙˙˙˙˙˙˙˙
.

active_std 

active_std˙˙˙˙˙˙˙˙˙
:
bwd_blk_rate_avg&#
bwd_blk_rate_avg˙˙˙˙˙˙˙˙˙
6
bwd_byts_b_avg$!
bwd_byts_b_avg˙˙˙˙˙˙˙˙˙
6
bwd_header_len$!
bwd_header_len˙˙˙˙˙˙˙˙˙
0
bwd_iat_max!
bwd_iat_max˙˙˙˙˙˙˙˙˙
2
bwd_iat_mean"
bwd_iat_mean˙˙˙˙˙˙˙˙˙
0
bwd_iat_min!
bwd_iat_min˙˙˙˙˙˙˙˙˙
0
bwd_iat_std!
bwd_iat_std˙˙˙˙˙˙˙˙˙
0
bwd_iat_tot!
bwd_iat_tot˙˙˙˙˙˙˙˙˙
8
bwd_pkt_len_max%"
bwd_pkt_len_max˙˙˙˙˙˙˙˙˙
:
bwd_pkt_len_mean&#
bwd_pkt_len_mean˙˙˙˙˙˙˙˙˙
8
bwd_pkt_len_min%"
bwd_pkt_len_min˙˙˙˙˙˙˙˙˙
8
bwd_pkt_len_std%"
bwd_pkt_len_std˙˙˙˙˙˙˙˙˙
6
bwd_pkts_b_avg$!
bwd_pkts_b_avg˙˙˙˙˙˙˙˙˙
.

bwd_pkts_s 

bwd_pkts_s˙˙˙˙˙˙˙˙˙
4
bwd_psh_flags# 
bwd_psh_flags˙˙˙˙˙˙˙˙˙
4
bwd_urg_flags# 
bwd_urg_flags˙˙˙˙˙˙˙˙˙
4
down_up_ratio# 
down_up_ratio˙˙˙˙˙˙˙˙˙
&
dst_ip
dst_ip˙˙˙˙˙˙˙˙˙	
*
dst_port
dst_port˙˙˙˙˙˙˙˙˙
2
ece_flag_cnt"
ece_flag_cnt˙˙˙˙˙˙˙˙˙
2
fin_flag_cnt"
fin_flag_cnt˙˙˙˙˙˙˙˙˙
0
flow_byts_s!
flow_byts_s˙˙˙˙˙˙˙˙˙
4
flow_duration# 
flow_duration˙˙˙˙˙˙˙˙˙
2
flow_iat_max"
flow_iat_max˙˙˙˙˙˙˙˙˙
4
flow_iat_mean# 
flow_iat_mean˙˙˙˙˙˙˙˙˙
2
flow_iat_min"
flow_iat_min˙˙˙˙˙˙˙˙˙
2
flow_iat_std"
flow_iat_std˙˙˙˙˙˙˙˙˙
0
flow_pkts_s!
flow_pkts_s˙˙˙˙˙˙˙˙˙
<
fwd_act_data_pkts'$
fwd_act_data_pkts˙˙˙˙˙˙˙˙˙
:
fwd_blk_rate_avg&#
fwd_blk_rate_avg˙˙˙˙˙˙˙˙˙
6
fwd_byts_b_avg$!
fwd_byts_b_avg˙˙˙˙˙˙˙˙˙
6
fwd_header_len$!
fwd_header_len˙˙˙˙˙˙˙˙˙
0
fwd_iat_max!
fwd_iat_max˙˙˙˙˙˙˙˙˙
2
fwd_iat_mean"
fwd_iat_mean˙˙˙˙˙˙˙˙˙
0
fwd_iat_min!
fwd_iat_min˙˙˙˙˙˙˙˙˙
0
fwd_iat_std!
fwd_iat_std˙˙˙˙˙˙˙˙˙
0
fwd_iat_tot!
fwd_iat_tot˙˙˙˙˙˙˙˙˙
8
fwd_pkt_len_max%"
fwd_pkt_len_max˙˙˙˙˙˙˙˙˙
:
fwd_pkt_len_mean&#
fwd_pkt_len_mean˙˙˙˙˙˙˙˙˙
8
fwd_pkt_len_min%"
fwd_pkt_len_min˙˙˙˙˙˙˙˙˙
8
fwd_pkt_len_std%"
fwd_pkt_len_std˙˙˙˙˙˙˙˙˙
6
fwd_pkts_b_avg$!
fwd_pkts_b_avg˙˙˙˙˙˙˙˙˙
.

fwd_pkts_s 

fwd_pkts_s˙˙˙˙˙˙˙˙˙
4
fwd_psh_flags# 
fwd_psh_flags˙˙˙˙˙˙˙˙˙
:
fwd_seg_size_min&#
fwd_seg_size_min˙˙˙˙˙˙˙˙˙
4
fwd_urg_flags# 
fwd_urg_flags˙˙˙˙˙˙˙˙˙
*
idle_max
idle_max˙˙˙˙˙˙˙˙˙
,
	idle_mean
	idle_mean˙˙˙˙˙˙˙˙˙
*
idle_min
idle_min˙˙˙˙˙˙˙˙˙
*
idle_std
idle_std˙˙˙˙˙˙˙˙˙
"
info
info˙˙˙˙˙˙˙˙˙
<
init_bwd_win_byts'$
init_bwd_win_byts˙˙˙˙˙˙˙˙˙
<
init_fwd_win_byts'$
init_fwd_win_byts˙˙˙˙˙˙˙˙˙
0
pkt_len_max!
pkt_len_max˙˙˙˙˙˙˙˙˙
2
pkt_len_mean"
pkt_len_mean˙˙˙˙˙˙˙˙˙
0
pkt_len_min!
pkt_len_min˙˙˙˙˙˙˙˙˙
0
pkt_len_std!
pkt_len_std˙˙˙˙˙˙˙˙˙
0
pkt_len_var!
pkt_len_var˙˙˙˙˙˙˙˙˙
.

pkt_length 

pkt_length˙˙˙˙˙˙˙˙˙
2
pkt_size_avg"
pkt_size_avg˙˙˙˙˙˙˙˙˙
*
protocol
protocol˙˙˙˙˙˙˙˙˙
2
psh_flag_cnt"
psh_flag_cnt˙˙˙˙˙˙˙˙˙
2
rst_flag_cnt"
rst_flag_cnt˙˙˙˙˙˙˙˙˙
&
src_ip
src_ip˙˙˙˙˙˙˙˙˙	
*
src_port
src_port˙˙˙˙˙˙˙˙˙
2
syn_flag_cnt"
syn_flag_cnt˙˙˙˙˙˙˙˙˙
,
	timestamp
	timestamp˙˙˙˙˙˙˙˙˙
2
tot_bwd_pkts"
tot_bwd_pkts˙˙˙˙˙˙˙˙˙
2
tot_fwd_pkts"
tot_fwd_pkts˙˙˙˙˙˙˙˙˙
8
totlen_bwd_pkts%"
totlen_bwd_pkts˙˙˙˙˙˙˙˙˙
8
totlen_fwd_pkts%"
totlen_fwd_pkts˙˙˙˙˙˙˙˙˙
2
urg_flag_cnt"
urg_flag_cnt˙˙˙˙˙˙˙˙˙
p
Ş "!
unknown˙˙˙˙˙˙˙˙˙ż 
0__inference_RandomForestModel_layer_call_fn_3813 -á˘Ý
Ő˘Ń
ĘŞĆ

No
No˙˙˙˙˙˙˙˙˙
2
ack_flag_cnt"
ack_flag_cnt˙˙˙˙˙˙˙˙˙
.

active_max 

active_max˙˙˙˙˙˙˙˙˙
0
active_mean!
active_mean˙˙˙˙˙˙˙˙˙
.

active_min 

active_min˙˙˙˙˙˙˙˙˙
.

active_std 

active_std˙˙˙˙˙˙˙˙˙
:
bwd_blk_rate_avg&#
bwd_blk_rate_avg˙˙˙˙˙˙˙˙˙
6
bwd_byts_b_avg$!
bwd_byts_b_avg˙˙˙˙˙˙˙˙˙
6
bwd_header_len$!
bwd_header_len˙˙˙˙˙˙˙˙˙
0
bwd_iat_max!
bwd_iat_max˙˙˙˙˙˙˙˙˙
2
bwd_iat_mean"
bwd_iat_mean˙˙˙˙˙˙˙˙˙
0
bwd_iat_min!
bwd_iat_min˙˙˙˙˙˙˙˙˙
0
bwd_iat_std!
bwd_iat_std˙˙˙˙˙˙˙˙˙
0
bwd_iat_tot!
bwd_iat_tot˙˙˙˙˙˙˙˙˙
8
bwd_pkt_len_max%"
bwd_pkt_len_max˙˙˙˙˙˙˙˙˙
:
bwd_pkt_len_mean&#
bwd_pkt_len_mean˙˙˙˙˙˙˙˙˙
8
bwd_pkt_len_min%"
bwd_pkt_len_min˙˙˙˙˙˙˙˙˙
8
bwd_pkt_len_std%"
bwd_pkt_len_std˙˙˙˙˙˙˙˙˙
6
bwd_pkts_b_avg$!
bwd_pkts_b_avg˙˙˙˙˙˙˙˙˙
.

bwd_pkts_s 

bwd_pkts_s˙˙˙˙˙˙˙˙˙
4
bwd_psh_flags# 
bwd_psh_flags˙˙˙˙˙˙˙˙˙
4
bwd_urg_flags# 
bwd_urg_flags˙˙˙˙˙˙˙˙˙
4
down_up_ratio# 
down_up_ratio˙˙˙˙˙˙˙˙˙
&
dst_ip
dst_ip˙˙˙˙˙˙˙˙˙	
*
dst_port
dst_port˙˙˙˙˙˙˙˙˙
2
ece_flag_cnt"
ece_flag_cnt˙˙˙˙˙˙˙˙˙
2
fin_flag_cnt"
fin_flag_cnt˙˙˙˙˙˙˙˙˙
0
flow_byts_s!
flow_byts_s˙˙˙˙˙˙˙˙˙
4
flow_duration# 
flow_duration˙˙˙˙˙˙˙˙˙
2
flow_iat_max"
flow_iat_max˙˙˙˙˙˙˙˙˙
4
flow_iat_mean# 
flow_iat_mean˙˙˙˙˙˙˙˙˙
2
flow_iat_min"
flow_iat_min˙˙˙˙˙˙˙˙˙
2
flow_iat_std"
flow_iat_std˙˙˙˙˙˙˙˙˙
0
flow_pkts_s!
flow_pkts_s˙˙˙˙˙˙˙˙˙
<
fwd_act_data_pkts'$
fwd_act_data_pkts˙˙˙˙˙˙˙˙˙
:
fwd_blk_rate_avg&#
fwd_blk_rate_avg˙˙˙˙˙˙˙˙˙
6
fwd_byts_b_avg$!
fwd_byts_b_avg˙˙˙˙˙˙˙˙˙
6
fwd_header_len$!
fwd_header_len˙˙˙˙˙˙˙˙˙
0
fwd_iat_max!
fwd_iat_max˙˙˙˙˙˙˙˙˙
2
fwd_iat_mean"
fwd_iat_mean˙˙˙˙˙˙˙˙˙
0
fwd_iat_min!
fwd_iat_min˙˙˙˙˙˙˙˙˙
0
fwd_iat_std!
fwd_iat_std˙˙˙˙˙˙˙˙˙
0
fwd_iat_tot!
fwd_iat_tot˙˙˙˙˙˙˙˙˙
8
fwd_pkt_len_max%"
fwd_pkt_len_max˙˙˙˙˙˙˙˙˙
:
fwd_pkt_len_mean&#
fwd_pkt_len_mean˙˙˙˙˙˙˙˙˙
8
fwd_pkt_len_min%"
fwd_pkt_len_min˙˙˙˙˙˙˙˙˙
8
fwd_pkt_len_std%"
fwd_pkt_len_std˙˙˙˙˙˙˙˙˙
6
fwd_pkts_b_avg$!
fwd_pkts_b_avg˙˙˙˙˙˙˙˙˙
.

fwd_pkts_s 

fwd_pkts_s˙˙˙˙˙˙˙˙˙
4
fwd_psh_flags# 
fwd_psh_flags˙˙˙˙˙˙˙˙˙
:
fwd_seg_size_min&#
fwd_seg_size_min˙˙˙˙˙˙˙˙˙
4
fwd_urg_flags# 
fwd_urg_flags˙˙˙˙˙˙˙˙˙
*
idle_max
idle_max˙˙˙˙˙˙˙˙˙
,
	idle_mean
	idle_mean˙˙˙˙˙˙˙˙˙
*
idle_min
idle_min˙˙˙˙˙˙˙˙˙
*
idle_std
idle_std˙˙˙˙˙˙˙˙˙
"
info
info˙˙˙˙˙˙˙˙˙
<
init_bwd_win_byts'$
init_bwd_win_byts˙˙˙˙˙˙˙˙˙
<
init_fwd_win_byts'$
init_fwd_win_byts˙˙˙˙˙˙˙˙˙
0
pkt_len_max!
pkt_len_max˙˙˙˙˙˙˙˙˙
2
pkt_len_mean"
pkt_len_mean˙˙˙˙˙˙˙˙˙
0
pkt_len_min!
pkt_len_min˙˙˙˙˙˙˙˙˙
0
pkt_len_std!
pkt_len_std˙˙˙˙˙˙˙˙˙
0
pkt_len_var!
pkt_len_var˙˙˙˙˙˙˙˙˙
.

pkt_length 

pkt_length˙˙˙˙˙˙˙˙˙
2
pkt_size_avg"
pkt_size_avg˙˙˙˙˙˙˙˙˙
*
protocol
protocol˙˙˙˙˙˙˙˙˙
2
psh_flag_cnt"
psh_flag_cnt˙˙˙˙˙˙˙˙˙
2
rst_flag_cnt"
rst_flag_cnt˙˙˙˙˙˙˙˙˙
&
src_ip
src_ip˙˙˙˙˙˙˙˙˙	
*
src_port
src_port˙˙˙˙˙˙˙˙˙
2
syn_flag_cnt"
syn_flag_cnt˙˙˙˙˙˙˙˙˙
,
	timestamp
	timestamp˙˙˙˙˙˙˙˙˙
2
tot_bwd_pkts"
tot_bwd_pkts˙˙˙˙˙˙˙˙˙
2
tot_fwd_pkts"
tot_fwd_pkts˙˙˙˙˙˙˙˙˙
8
totlen_bwd_pkts%"
totlen_bwd_pkts˙˙˙˙˙˙˙˙˙
8
totlen_fwd_pkts%"
totlen_fwd_pkts˙˙˙˙˙˙˙˙˙
2
urg_flag_cnt"
urg_flag_cnt˙˙˙˙˙˙˙˙˙
p 
Ş "!
unknown˙˙˙˙˙˙˙˙˙ßB
)__inference__build_normalized_inputs_3986ąB˙#˘ű#
ó#˘ď#
ě#Şč#
%
No
	inputs_no˙˙˙˙˙˙˙˙˙
9
ack_flag_cnt)&
inputs_ack_flag_cnt˙˙˙˙˙˙˙˙˙
5

active_max'$
inputs_active_max˙˙˙˙˙˙˙˙˙
7
active_mean(%
inputs_active_mean˙˙˙˙˙˙˙˙˙
5

active_min'$
inputs_active_min˙˙˙˙˙˙˙˙˙
5

active_std'$
inputs_active_std˙˙˙˙˙˙˙˙˙
A
bwd_blk_rate_avg-*
inputs_bwd_blk_rate_avg˙˙˙˙˙˙˙˙˙
=
bwd_byts_b_avg+(
inputs_bwd_byts_b_avg˙˙˙˙˙˙˙˙˙
=
bwd_header_len+(
inputs_bwd_header_len˙˙˙˙˙˙˙˙˙
7
bwd_iat_max(%
inputs_bwd_iat_max˙˙˙˙˙˙˙˙˙
9
bwd_iat_mean)&
inputs_bwd_iat_mean˙˙˙˙˙˙˙˙˙
7
bwd_iat_min(%
inputs_bwd_iat_min˙˙˙˙˙˙˙˙˙
7
bwd_iat_std(%
inputs_bwd_iat_std˙˙˙˙˙˙˙˙˙
7
bwd_iat_tot(%
inputs_bwd_iat_tot˙˙˙˙˙˙˙˙˙
?
bwd_pkt_len_max,)
inputs_bwd_pkt_len_max˙˙˙˙˙˙˙˙˙
A
bwd_pkt_len_mean-*
inputs_bwd_pkt_len_mean˙˙˙˙˙˙˙˙˙
?
bwd_pkt_len_min,)
inputs_bwd_pkt_len_min˙˙˙˙˙˙˙˙˙
?
bwd_pkt_len_std,)
inputs_bwd_pkt_len_std˙˙˙˙˙˙˙˙˙
=
bwd_pkts_b_avg+(
inputs_bwd_pkts_b_avg˙˙˙˙˙˙˙˙˙
5

bwd_pkts_s'$
inputs_bwd_pkts_s˙˙˙˙˙˙˙˙˙
;
bwd_psh_flags*'
inputs_bwd_psh_flags˙˙˙˙˙˙˙˙˙
;
bwd_urg_flags*'
inputs_bwd_urg_flags˙˙˙˙˙˙˙˙˙
;
down_up_ratio*'
inputs_down_up_ratio˙˙˙˙˙˙˙˙˙
-
dst_ip# 
inputs_dst_ip˙˙˙˙˙˙˙˙˙	
1
dst_port%"
inputs_dst_port˙˙˙˙˙˙˙˙˙
9
ece_flag_cnt)&
inputs_ece_flag_cnt˙˙˙˙˙˙˙˙˙
9
fin_flag_cnt)&
inputs_fin_flag_cnt˙˙˙˙˙˙˙˙˙
7
flow_byts_s(%
inputs_flow_byts_s˙˙˙˙˙˙˙˙˙
;
flow_duration*'
inputs_flow_duration˙˙˙˙˙˙˙˙˙
9
flow_iat_max)&
inputs_flow_iat_max˙˙˙˙˙˙˙˙˙
;
flow_iat_mean*'
inputs_flow_iat_mean˙˙˙˙˙˙˙˙˙
9
flow_iat_min)&
inputs_flow_iat_min˙˙˙˙˙˙˙˙˙
9
flow_iat_std)&
inputs_flow_iat_std˙˙˙˙˙˙˙˙˙
7
flow_pkts_s(%
inputs_flow_pkts_s˙˙˙˙˙˙˙˙˙
C
fwd_act_data_pkts.+
inputs_fwd_act_data_pkts˙˙˙˙˙˙˙˙˙
A
fwd_blk_rate_avg-*
inputs_fwd_blk_rate_avg˙˙˙˙˙˙˙˙˙
=
fwd_byts_b_avg+(
inputs_fwd_byts_b_avg˙˙˙˙˙˙˙˙˙
=
fwd_header_len+(
inputs_fwd_header_len˙˙˙˙˙˙˙˙˙
7
fwd_iat_max(%
inputs_fwd_iat_max˙˙˙˙˙˙˙˙˙
9
fwd_iat_mean)&
inputs_fwd_iat_mean˙˙˙˙˙˙˙˙˙
7
fwd_iat_min(%
inputs_fwd_iat_min˙˙˙˙˙˙˙˙˙
7
fwd_iat_std(%
inputs_fwd_iat_std˙˙˙˙˙˙˙˙˙
7
fwd_iat_tot(%
inputs_fwd_iat_tot˙˙˙˙˙˙˙˙˙
?
fwd_pkt_len_max,)
inputs_fwd_pkt_len_max˙˙˙˙˙˙˙˙˙
A
fwd_pkt_len_mean-*
inputs_fwd_pkt_len_mean˙˙˙˙˙˙˙˙˙
?
fwd_pkt_len_min,)
inputs_fwd_pkt_len_min˙˙˙˙˙˙˙˙˙
?
fwd_pkt_len_std,)
inputs_fwd_pkt_len_std˙˙˙˙˙˙˙˙˙
=
fwd_pkts_b_avg+(
inputs_fwd_pkts_b_avg˙˙˙˙˙˙˙˙˙
5

fwd_pkts_s'$
inputs_fwd_pkts_s˙˙˙˙˙˙˙˙˙
;
fwd_psh_flags*'
inputs_fwd_psh_flags˙˙˙˙˙˙˙˙˙
A
fwd_seg_size_min-*
inputs_fwd_seg_size_min˙˙˙˙˙˙˙˙˙
;
fwd_urg_flags*'
inputs_fwd_urg_flags˙˙˙˙˙˙˙˙˙
1
idle_max%"
inputs_idle_max˙˙˙˙˙˙˙˙˙
3
	idle_mean&#
inputs_idle_mean˙˙˙˙˙˙˙˙˙
1
idle_min%"
inputs_idle_min˙˙˙˙˙˙˙˙˙
1
idle_std%"
inputs_idle_std˙˙˙˙˙˙˙˙˙
)
info!
inputs_info˙˙˙˙˙˙˙˙˙
C
init_bwd_win_byts.+
inputs_init_bwd_win_byts˙˙˙˙˙˙˙˙˙
C
init_fwd_win_byts.+
inputs_init_fwd_win_byts˙˙˙˙˙˙˙˙˙
7
pkt_len_max(%
inputs_pkt_len_max˙˙˙˙˙˙˙˙˙
9
pkt_len_mean)&
inputs_pkt_len_mean˙˙˙˙˙˙˙˙˙
7
pkt_len_min(%
inputs_pkt_len_min˙˙˙˙˙˙˙˙˙
7
pkt_len_std(%
inputs_pkt_len_std˙˙˙˙˙˙˙˙˙
7
pkt_len_var(%
inputs_pkt_len_var˙˙˙˙˙˙˙˙˙
5

pkt_length'$
inputs_pkt_length˙˙˙˙˙˙˙˙˙
9
pkt_size_avg)&
inputs_pkt_size_avg˙˙˙˙˙˙˙˙˙
1
protocol%"
inputs_protocol˙˙˙˙˙˙˙˙˙
9
psh_flag_cnt)&
inputs_psh_flag_cnt˙˙˙˙˙˙˙˙˙
9
rst_flag_cnt)&
inputs_rst_flag_cnt˙˙˙˙˙˙˙˙˙
-
src_ip# 
inputs_src_ip˙˙˙˙˙˙˙˙˙	
1
src_port%"
inputs_src_port˙˙˙˙˙˙˙˙˙
9
syn_flag_cnt)&
inputs_syn_flag_cnt˙˙˙˙˙˙˙˙˙
3
	timestamp&#
inputs_timestamp˙˙˙˙˙˙˙˙˙
9
tot_bwd_pkts)&
inputs_tot_bwd_pkts˙˙˙˙˙˙˙˙˙
9
tot_fwd_pkts)&
inputs_tot_fwd_pkts˙˙˙˙˙˙˙˙˙
?
totlen_bwd_pkts,)
inputs_totlen_bwd_pkts˙˙˙˙˙˙˙˙˙
?
totlen_fwd_pkts,)
inputs_totlen_fwd_pkts˙˙˙˙˙˙˙˙˙
9
urg_flag_cnt)&
inputs_urg_flag_cnt˙˙˙˙˙˙˙˙˙
Ş "ŹŞ¨
2
ack_flag_cnt"
ack_flag_cnt˙˙˙˙˙˙˙˙˙
.

active_max 

active_max˙˙˙˙˙˙˙˙˙
0
active_mean!
active_mean˙˙˙˙˙˙˙˙˙
.

active_min 

active_min˙˙˙˙˙˙˙˙˙
.

active_std 

active_std˙˙˙˙˙˙˙˙˙
:
bwd_blk_rate_avg&#
bwd_blk_rate_avg˙˙˙˙˙˙˙˙˙
6
bwd_byts_b_avg$!
bwd_byts_b_avg˙˙˙˙˙˙˙˙˙
6
bwd_header_len$!
bwd_header_len˙˙˙˙˙˙˙˙˙
0
bwd_iat_max!
bwd_iat_max˙˙˙˙˙˙˙˙˙
2
bwd_iat_mean"
bwd_iat_mean˙˙˙˙˙˙˙˙˙
0
bwd_iat_min!
bwd_iat_min˙˙˙˙˙˙˙˙˙
0
bwd_iat_std!
bwd_iat_std˙˙˙˙˙˙˙˙˙
0
bwd_iat_tot!
bwd_iat_tot˙˙˙˙˙˙˙˙˙
8
bwd_pkt_len_max%"
bwd_pkt_len_max˙˙˙˙˙˙˙˙˙
:
bwd_pkt_len_mean&#
bwd_pkt_len_mean˙˙˙˙˙˙˙˙˙
8
bwd_pkt_len_min%"
bwd_pkt_len_min˙˙˙˙˙˙˙˙˙
8
bwd_pkt_len_std%"
bwd_pkt_len_std˙˙˙˙˙˙˙˙˙
6
bwd_pkts_b_avg$!
bwd_pkts_b_avg˙˙˙˙˙˙˙˙˙
.

bwd_pkts_s 

bwd_pkts_s˙˙˙˙˙˙˙˙˙
4
bwd_psh_flags# 
bwd_psh_flags˙˙˙˙˙˙˙˙˙
4
bwd_urg_flags# 
bwd_urg_flags˙˙˙˙˙˙˙˙˙
4
down_up_ratio# 
down_up_ratio˙˙˙˙˙˙˙˙˙
*
dst_port
dst_port˙˙˙˙˙˙˙˙˙
2
ece_flag_cnt"
ece_flag_cnt˙˙˙˙˙˙˙˙˙
2
fin_flag_cnt"
fin_flag_cnt˙˙˙˙˙˙˙˙˙
0
flow_byts_s!
flow_byts_s˙˙˙˙˙˙˙˙˙
4
flow_duration# 
flow_duration˙˙˙˙˙˙˙˙˙
2
flow_iat_max"
flow_iat_max˙˙˙˙˙˙˙˙˙
4
flow_iat_mean# 
flow_iat_mean˙˙˙˙˙˙˙˙˙
2
flow_iat_min"
flow_iat_min˙˙˙˙˙˙˙˙˙
2
flow_iat_std"
flow_iat_std˙˙˙˙˙˙˙˙˙
0
flow_pkts_s!
flow_pkts_s˙˙˙˙˙˙˙˙˙
<
fwd_act_data_pkts'$
fwd_act_data_pkts˙˙˙˙˙˙˙˙˙
:
fwd_blk_rate_avg&#
fwd_blk_rate_avg˙˙˙˙˙˙˙˙˙
6
fwd_byts_b_avg$!
fwd_byts_b_avg˙˙˙˙˙˙˙˙˙
6
fwd_header_len$!
fwd_header_len˙˙˙˙˙˙˙˙˙
0
fwd_iat_max!
fwd_iat_max˙˙˙˙˙˙˙˙˙
2
fwd_iat_mean"
fwd_iat_mean˙˙˙˙˙˙˙˙˙
0
fwd_iat_min!
fwd_iat_min˙˙˙˙˙˙˙˙˙
0
fwd_iat_std!
fwd_iat_std˙˙˙˙˙˙˙˙˙
0
fwd_iat_tot!
fwd_iat_tot˙˙˙˙˙˙˙˙˙
8
fwd_pkt_len_max%"
fwd_pkt_len_max˙˙˙˙˙˙˙˙˙
:
fwd_pkt_len_mean&#
fwd_pkt_len_mean˙˙˙˙˙˙˙˙˙
8
fwd_pkt_len_min%"
fwd_pkt_len_min˙˙˙˙˙˙˙˙˙
8
fwd_pkt_len_std%"
fwd_pkt_len_std˙˙˙˙˙˙˙˙˙
6
fwd_pkts_b_avg$!
fwd_pkts_b_avg˙˙˙˙˙˙˙˙˙
.

fwd_pkts_s 

fwd_pkts_s˙˙˙˙˙˙˙˙˙
4
fwd_psh_flags# 
fwd_psh_flags˙˙˙˙˙˙˙˙˙
:
fwd_seg_size_min&#
fwd_seg_size_min˙˙˙˙˙˙˙˙˙
4
fwd_urg_flags# 
fwd_urg_flags˙˙˙˙˙˙˙˙˙
*
idle_max
idle_max˙˙˙˙˙˙˙˙˙
,
	idle_mean
	idle_mean˙˙˙˙˙˙˙˙˙
*
idle_min
idle_min˙˙˙˙˙˙˙˙˙
*
idle_std
idle_std˙˙˙˙˙˙˙˙˙
"
info
info˙˙˙˙˙˙˙˙˙
<
init_bwd_win_byts'$
init_bwd_win_byts˙˙˙˙˙˙˙˙˙
<
init_fwd_win_byts'$
init_fwd_win_byts˙˙˙˙˙˙˙˙˙
0
pkt_len_max!
pkt_len_max˙˙˙˙˙˙˙˙˙
2
pkt_len_mean"
pkt_len_mean˙˙˙˙˙˙˙˙˙
0
pkt_len_min!
pkt_len_min˙˙˙˙˙˙˙˙˙
0
pkt_len_std!
pkt_len_std˙˙˙˙˙˙˙˙˙
0
pkt_len_var!
pkt_len_var˙˙˙˙˙˙˙˙˙
.

pkt_length 

pkt_length˙˙˙˙˙˙˙˙˙
2
pkt_size_avg"
pkt_size_avg˙˙˙˙˙˙˙˙˙
*
protocol
protocol˙˙˙˙˙˙˙˙˙
2
psh_flag_cnt"
psh_flag_cnt˙˙˙˙˙˙˙˙˙
2
rst_flag_cnt"
rst_flag_cnt˙˙˙˙˙˙˙˙˙
*
src_port
src_port˙˙˙˙˙˙˙˙˙
2
syn_flag_cnt"
syn_flag_cnt˙˙˙˙˙˙˙˙˙
2
tot_bwd_pkts"
tot_bwd_pkts˙˙˙˙˙˙˙˙˙
2
tot_fwd_pkts"
tot_fwd_pkts˙˙˙˙˙˙˙˙˙
8
totlen_bwd_pkts%"
totlen_bwd_pkts˙˙˙˙˙˙˙˙˙
8
totlen_fwd_pkts%"
totlen_fwd_pkts˙˙˙˙˙˙˙˙˙
2
urg_flag_cnt"
urg_flag_cnt˙˙˙˙˙˙˙˙˙>
__inference__creator_4254!˘

˘ 
Ş "
unknown @
__inference__destroyer_4265!˘

˘ 
Ş "
unknown 
&__inference__finalize_predictions_3995ďÉ˘Ĺ
˝˘š
`
Ž˛Ş
ModelOutputL
dense_predictions74
predictions_dense_predictions˙˙˙˙˙˙˙˙˙M
dense_col_representation1.
$predictions_dense_col_representation
p 
Ş "!
unknown˙˙˙˙˙˙˙˙˙F
__inference__initializer_4261%.-˘

˘ 
Ş "
unknown ź 
__inference__wrapped_model_3315 -Ý˘Ů
Ń˘Í
ĘŞĆ

No
No˙˙˙˙˙˙˙˙˙
2
ack_flag_cnt"
ack_flag_cnt˙˙˙˙˙˙˙˙˙
.

active_max 

active_max˙˙˙˙˙˙˙˙˙
0
active_mean!
active_mean˙˙˙˙˙˙˙˙˙
.

active_min 

active_min˙˙˙˙˙˙˙˙˙
.

active_std 

active_std˙˙˙˙˙˙˙˙˙
:
bwd_blk_rate_avg&#
bwd_blk_rate_avg˙˙˙˙˙˙˙˙˙
6
bwd_byts_b_avg$!
bwd_byts_b_avg˙˙˙˙˙˙˙˙˙
6
bwd_header_len$!
bwd_header_len˙˙˙˙˙˙˙˙˙
0
bwd_iat_max!
bwd_iat_max˙˙˙˙˙˙˙˙˙
2
bwd_iat_mean"
bwd_iat_mean˙˙˙˙˙˙˙˙˙
0
bwd_iat_min!
bwd_iat_min˙˙˙˙˙˙˙˙˙
0
bwd_iat_std!
bwd_iat_std˙˙˙˙˙˙˙˙˙
0
bwd_iat_tot!
bwd_iat_tot˙˙˙˙˙˙˙˙˙
8
bwd_pkt_len_max%"
bwd_pkt_len_max˙˙˙˙˙˙˙˙˙
:
bwd_pkt_len_mean&#
bwd_pkt_len_mean˙˙˙˙˙˙˙˙˙
8
bwd_pkt_len_min%"
bwd_pkt_len_min˙˙˙˙˙˙˙˙˙
8
bwd_pkt_len_std%"
bwd_pkt_len_std˙˙˙˙˙˙˙˙˙
6
bwd_pkts_b_avg$!
bwd_pkts_b_avg˙˙˙˙˙˙˙˙˙
.

bwd_pkts_s 

bwd_pkts_s˙˙˙˙˙˙˙˙˙
4
bwd_psh_flags# 
bwd_psh_flags˙˙˙˙˙˙˙˙˙
4
bwd_urg_flags# 
bwd_urg_flags˙˙˙˙˙˙˙˙˙
4
down_up_ratio# 
down_up_ratio˙˙˙˙˙˙˙˙˙
&
dst_ip
dst_ip˙˙˙˙˙˙˙˙˙	
*
dst_port
dst_port˙˙˙˙˙˙˙˙˙
2
ece_flag_cnt"
ece_flag_cnt˙˙˙˙˙˙˙˙˙
2
fin_flag_cnt"
fin_flag_cnt˙˙˙˙˙˙˙˙˙
0
flow_byts_s!
flow_byts_s˙˙˙˙˙˙˙˙˙
4
flow_duration# 
flow_duration˙˙˙˙˙˙˙˙˙
2
flow_iat_max"
flow_iat_max˙˙˙˙˙˙˙˙˙
4
flow_iat_mean# 
flow_iat_mean˙˙˙˙˙˙˙˙˙
2
flow_iat_min"
flow_iat_min˙˙˙˙˙˙˙˙˙
2
flow_iat_std"
flow_iat_std˙˙˙˙˙˙˙˙˙
0
flow_pkts_s!
flow_pkts_s˙˙˙˙˙˙˙˙˙
<
fwd_act_data_pkts'$
fwd_act_data_pkts˙˙˙˙˙˙˙˙˙
:
fwd_blk_rate_avg&#
fwd_blk_rate_avg˙˙˙˙˙˙˙˙˙
6
fwd_byts_b_avg$!
fwd_byts_b_avg˙˙˙˙˙˙˙˙˙
6
fwd_header_len$!
fwd_header_len˙˙˙˙˙˙˙˙˙
0
fwd_iat_max!
fwd_iat_max˙˙˙˙˙˙˙˙˙
2
fwd_iat_mean"
fwd_iat_mean˙˙˙˙˙˙˙˙˙
0
fwd_iat_min!
fwd_iat_min˙˙˙˙˙˙˙˙˙
0
fwd_iat_std!
fwd_iat_std˙˙˙˙˙˙˙˙˙
0
fwd_iat_tot!
fwd_iat_tot˙˙˙˙˙˙˙˙˙
8
fwd_pkt_len_max%"
fwd_pkt_len_max˙˙˙˙˙˙˙˙˙
:
fwd_pkt_len_mean&#
fwd_pkt_len_mean˙˙˙˙˙˙˙˙˙
8
fwd_pkt_len_min%"
fwd_pkt_len_min˙˙˙˙˙˙˙˙˙
8
fwd_pkt_len_std%"
fwd_pkt_len_std˙˙˙˙˙˙˙˙˙
6
fwd_pkts_b_avg$!
fwd_pkts_b_avg˙˙˙˙˙˙˙˙˙
.

fwd_pkts_s 

fwd_pkts_s˙˙˙˙˙˙˙˙˙
4
fwd_psh_flags# 
fwd_psh_flags˙˙˙˙˙˙˙˙˙
:
fwd_seg_size_min&#
fwd_seg_size_min˙˙˙˙˙˙˙˙˙
4
fwd_urg_flags# 
fwd_urg_flags˙˙˙˙˙˙˙˙˙
*
idle_max
idle_max˙˙˙˙˙˙˙˙˙
,
	idle_mean
	idle_mean˙˙˙˙˙˙˙˙˙
*
idle_min
idle_min˙˙˙˙˙˙˙˙˙
*
idle_std
idle_std˙˙˙˙˙˙˙˙˙
"
info
info˙˙˙˙˙˙˙˙˙
<
init_bwd_win_byts'$
init_bwd_win_byts˙˙˙˙˙˙˙˙˙
<
init_fwd_win_byts'$
init_fwd_win_byts˙˙˙˙˙˙˙˙˙
0
pkt_len_max!
pkt_len_max˙˙˙˙˙˙˙˙˙
2
pkt_len_mean"
pkt_len_mean˙˙˙˙˙˙˙˙˙
0
pkt_len_min!
pkt_len_min˙˙˙˙˙˙˙˙˙
0
pkt_len_std!
pkt_len_std˙˙˙˙˙˙˙˙˙
0
pkt_len_var!
pkt_len_var˙˙˙˙˙˙˙˙˙
.

pkt_length 

pkt_length˙˙˙˙˙˙˙˙˙
2
pkt_size_avg"
pkt_size_avg˙˙˙˙˙˙˙˙˙
*
protocol
protocol˙˙˙˙˙˙˙˙˙
2
psh_flag_cnt"
psh_flag_cnt˙˙˙˙˙˙˙˙˙
2
rst_flag_cnt"
rst_flag_cnt˙˙˙˙˙˙˙˙˙
&
src_ip
src_ip˙˙˙˙˙˙˙˙˙	
*
src_port
src_port˙˙˙˙˙˙˙˙˙
2
syn_flag_cnt"
syn_flag_cnt˙˙˙˙˙˙˙˙˙
,
	timestamp
	timestamp˙˙˙˙˙˙˙˙˙
2
tot_bwd_pkts"
tot_bwd_pkts˙˙˙˙˙˙˙˙˙
2
tot_fwd_pkts"
tot_fwd_pkts˙˙˙˙˙˙˙˙˙
8
totlen_bwd_pkts%"
totlen_bwd_pkts˙˙˙˙˙˙˙˙˙
8
totlen_fwd_pkts%"
totlen_fwd_pkts˙˙˙˙˙˙˙˙˙
2
urg_flag_cnt"
urg_flag_cnt˙˙˙˙˙˙˙˙˙
Ş "3Ş0
.
output_1"
output_1˙˙˙˙˙˙˙˙˙Ć$
__inference_call_4160Ź$-$˘˙#
÷#˘ó#
ě#Şč#
%
No
	inputs_no˙˙˙˙˙˙˙˙˙
9
ack_flag_cnt)&
inputs_ack_flag_cnt˙˙˙˙˙˙˙˙˙
5

active_max'$
inputs_active_max˙˙˙˙˙˙˙˙˙
7
active_mean(%
inputs_active_mean˙˙˙˙˙˙˙˙˙
5

active_min'$
inputs_active_min˙˙˙˙˙˙˙˙˙
5

active_std'$
inputs_active_std˙˙˙˙˙˙˙˙˙
A
bwd_blk_rate_avg-*
inputs_bwd_blk_rate_avg˙˙˙˙˙˙˙˙˙
=
bwd_byts_b_avg+(
inputs_bwd_byts_b_avg˙˙˙˙˙˙˙˙˙
=
bwd_header_len+(
inputs_bwd_header_len˙˙˙˙˙˙˙˙˙
7
bwd_iat_max(%
inputs_bwd_iat_max˙˙˙˙˙˙˙˙˙
9
bwd_iat_mean)&
inputs_bwd_iat_mean˙˙˙˙˙˙˙˙˙
7
bwd_iat_min(%
inputs_bwd_iat_min˙˙˙˙˙˙˙˙˙
7
bwd_iat_std(%
inputs_bwd_iat_std˙˙˙˙˙˙˙˙˙
7
bwd_iat_tot(%
inputs_bwd_iat_tot˙˙˙˙˙˙˙˙˙
?
bwd_pkt_len_max,)
inputs_bwd_pkt_len_max˙˙˙˙˙˙˙˙˙
A
bwd_pkt_len_mean-*
inputs_bwd_pkt_len_mean˙˙˙˙˙˙˙˙˙
?
bwd_pkt_len_min,)
inputs_bwd_pkt_len_min˙˙˙˙˙˙˙˙˙
?
bwd_pkt_len_std,)
inputs_bwd_pkt_len_std˙˙˙˙˙˙˙˙˙
=
bwd_pkts_b_avg+(
inputs_bwd_pkts_b_avg˙˙˙˙˙˙˙˙˙
5

bwd_pkts_s'$
inputs_bwd_pkts_s˙˙˙˙˙˙˙˙˙
;
bwd_psh_flags*'
inputs_bwd_psh_flags˙˙˙˙˙˙˙˙˙
;
bwd_urg_flags*'
inputs_bwd_urg_flags˙˙˙˙˙˙˙˙˙
;
down_up_ratio*'
inputs_down_up_ratio˙˙˙˙˙˙˙˙˙
-
dst_ip# 
inputs_dst_ip˙˙˙˙˙˙˙˙˙	
1
dst_port%"
inputs_dst_port˙˙˙˙˙˙˙˙˙
9
ece_flag_cnt)&
inputs_ece_flag_cnt˙˙˙˙˙˙˙˙˙
9
fin_flag_cnt)&
inputs_fin_flag_cnt˙˙˙˙˙˙˙˙˙
7
flow_byts_s(%
inputs_flow_byts_s˙˙˙˙˙˙˙˙˙
;
flow_duration*'
inputs_flow_duration˙˙˙˙˙˙˙˙˙
9
flow_iat_max)&
inputs_flow_iat_max˙˙˙˙˙˙˙˙˙
;
flow_iat_mean*'
inputs_flow_iat_mean˙˙˙˙˙˙˙˙˙
9
flow_iat_min)&
inputs_flow_iat_min˙˙˙˙˙˙˙˙˙
9
flow_iat_std)&
inputs_flow_iat_std˙˙˙˙˙˙˙˙˙
7
flow_pkts_s(%
inputs_flow_pkts_s˙˙˙˙˙˙˙˙˙
C
fwd_act_data_pkts.+
inputs_fwd_act_data_pkts˙˙˙˙˙˙˙˙˙
A
fwd_blk_rate_avg-*
inputs_fwd_blk_rate_avg˙˙˙˙˙˙˙˙˙
=
fwd_byts_b_avg+(
inputs_fwd_byts_b_avg˙˙˙˙˙˙˙˙˙
=
fwd_header_len+(
inputs_fwd_header_len˙˙˙˙˙˙˙˙˙
7
fwd_iat_max(%
inputs_fwd_iat_max˙˙˙˙˙˙˙˙˙
9
fwd_iat_mean)&
inputs_fwd_iat_mean˙˙˙˙˙˙˙˙˙
7
fwd_iat_min(%
inputs_fwd_iat_min˙˙˙˙˙˙˙˙˙
7
fwd_iat_std(%
inputs_fwd_iat_std˙˙˙˙˙˙˙˙˙
7
fwd_iat_tot(%
inputs_fwd_iat_tot˙˙˙˙˙˙˙˙˙
?
fwd_pkt_len_max,)
inputs_fwd_pkt_len_max˙˙˙˙˙˙˙˙˙
A
fwd_pkt_len_mean-*
inputs_fwd_pkt_len_mean˙˙˙˙˙˙˙˙˙
?
fwd_pkt_len_min,)
inputs_fwd_pkt_len_min˙˙˙˙˙˙˙˙˙
?
fwd_pkt_len_std,)
inputs_fwd_pkt_len_std˙˙˙˙˙˙˙˙˙
=
fwd_pkts_b_avg+(
inputs_fwd_pkts_b_avg˙˙˙˙˙˙˙˙˙
5

fwd_pkts_s'$
inputs_fwd_pkts_s˙˙˙˙˙˙˙˙˙
;
fwd_psh_flags*'
inputs_fwd_psh_flags˙˙˙˙˙˙˙˙˙
A
fwd_seg_size_min-*
inputs_fwd_seg_size_min˙˙˙˙˙˙˙˙˙
;
fwd_urg_flags*'
inputs_fwd_urg_flags˙˙˙˙˙˙˙˙˙
1
idle_max%"
inputs_idle_max˙˙˙˙˙˙˙˙˙
3
	idle_mean&#
inputs_idle_mean˙˙˙˙˙˙˙˙˙
1
idle_min%"
inputs_idle_min˙˙˙˙˙˙˙˙˙
1
idle_std%"
inputs_idle_std˙˙˙˙˙˙˙˙˙
)
info!
inputs_info˙˙˙˙˙˙˙˙˙
C
init_bwd_win_byts.+
inputs_init_bwd_win_byts˙˙˙˙˙˙˙˙˙
C
init_fwd_win_byts.+
inputs_init_fwd_win_byts˙˙˙˙˙˙˙˙˙
7
pkt_len_max(%
inputs_pkt_len_max˙˙˙˙˙˙˙˙˙
9
pkt_len_mean)&
inputs_pkt_len_mean˙˙˙˙˙˙˙˙˙
7
pkt_len_min(%
inputs_pkt_len_min˙˙˙˙˙˙˙˙˙
7
pkt_len_std(%
inputs_pkt_len_std˙˙˙˙˙˙˙˙˙
7
pkt_len_var(%
inputs_pkt_len_var˙˙˙˙˙˙˙˙˙
5

pkt_length'$
inputs_pkt_length˙˙˙˙˙˙˙˙˙
9
pkt_size_avg)&
inputs_pkt_size_avg˙˙˙˙˙˙˙˙˙
1
protocol%"
inputs_protocol˙˙˙˙˙˙˙˙˙
9
psh_flag_cnt)&
inputs_psh_flag_cnt˙˙˙˙˙˙˙˙˙
9
rst_flag_cnt)&
inputs_rst_flag_cnt˙˙˙˙˙˙˙˙˙
-
src_ip# 
inputs_src_ip˙˙˙˙˙˙˙˙˙	
1
src_port%"
inputs_src_port˙˙˙˙˙˙˙˙˙
9
syn_flag_cnt)&
inputs_syn_flag_cnt˙˙˙˙˙˙˙˙˙
3
	timestamp&#
inputs_timestamp˙˙˙˙˙˙˙˙˙
9
tot_bwd_pkts)&
inputs_tot_bwd_pkts˙˙˙˙˙˙˙˙˙
9
tot_fwd_pkts)&
inputs_tot_fwd_pkts˙˙˙˙˙˙˙˙˙
?
totlen_bwd_pkts,)
inputs_totlen_bwd_pkts˙˙˙˙˙˙˙˙˙
?
totlen_fwd_pkts,)
inputs_totlen_fwd_pkts˙˙˙˙˙˙˙˙˙
9
urg_flag_cnt)&
inputs_urg_flag_cnt˙˙˙˙˙˙˙˙˙
p 
Ş "!
unknown˙˙˙˙˙˙˙˙˙¸ 
"__inference_signature_wrapper_4250 -Ö˘Ň
˘ 
ĘŞĆ

No
no˙˙˙˙˙˙˙˙˙
2
ack_flag_cnt"
ack_flag_cnt˙˙˙˙˙˙˙˙˙
.

active_max 

active_max˙˙˙˙˙˙˙˙˙
0
active_mean!
active_mean˙˙˙˙˙˙˙˙˙
.

active_min 

active_min˙˙˙˙˙˙˙˙˙
.

active_std 

active_std˙˙˙˙˙˙˙˙˙
:
bwd_blk_rate_avg&#
bwd_blk_rate_avg˙˙˙˙˙˙˙˙˙
6
bwd_byts_b_avg$!
bwd_byts_b_avg˙˙˙˙˙˙˙˙˙
6
bwd_header_len$!
bwd_header_len˙˙˙˙˙˙˙˙˙
0
bwd_iat_max!
bwd_iat_max˙˙˙˙˙˙˙˙˙
2
bwd_iat_mean"
bwd_iat_mean˙˙˙˙˙˙˙˙˙
0
bwd_iat_min!
bwd_iat_min˙˙˙˙˙˙˙˙˙
0
bwd_iat_std!
bwd_iat_std˙˙˙˙˙˙˙˙˙
0
bwd_iat_tot!
bwd_iat_tot˙˙˙˙˙˙˙˙˙
8
bwd_pkt_len_max%"
bwd_pkt_len_max˙˙˙˙˙˙˙˙˙
:
bwd_pkt_len_mean&#
bwd_pkt_len_mean˙˙˙˙˙˙˙˙˙
8
bwd_pkt_len_min%"
bwd_pkt_len_min˙˙˙˙˙˙˙˙˙
8
bwd_pkt_len_std%"
bwd_pkt_len_std˙˙˙˙˙˙˙˙˙
6
bwd_pkts_b_avg$!
bwd_pkts_b_avg˙˙˙˙˙˙˙˙˙
.

bwd_pkts_s 

bwd_pkts_s˙˙˙˙˙˙˙˙˙
4
bwd_psh_flags# 
bwd_psh_flags˙˙˙˙˙˙˙˙˙
4
bwd_urg_flags# 
bwd_urg_flags˙˙˙˙˙˙˙˙˙
4
down_up_ratio# 
down_up_ratio˙˙˙˙˙˙˙˙˙
&
dst_ip
dst_ip˙˙˙˙˙˙˙˙˙	
*
dst_port
dst_port˙˙˙˙˙˙˙˙˙
2
ece_flag_cnt"
ece_flag_cnt˙˙˙˙˙˙˙˙˙
2
fin_flag_cnt"
fin_flag_cnt˙˙˙˙˙˙˙˙˙
0
flow_byts_s!
flow_byts_s˙˙˙˙˙˙˙˙˙
4
flow_duration# 
flow_duration˙˙˙˙˙˙˙˙˙
2
flow_iat_max"
flow_iat_max˙˙˙˙˙˙˙˙˙
4
flow_iat_mean# 
flow_iat_mean˙˙˙˙˙˙˙˙˙
2
flow_iat_min"
flow_iat_min˙˙˙˙˙˙˙˙˙
2
flow_iat_std"
flow_iat_std˙˙˙˙˙˙˙˙˙
0
flow_pkts_s!
flow_pkts_s˙˙˙˙˙˙˙˙˙
<
fwd_act_data_pkts'$
fwd_act_data_pkts˙˙˙˙˙˙˙˙˙
:
fwd_blk_rate_avg&#
fwd_blk_rate_avg˙˙˙˙˙˙˙˙˙
6
fwd_byts_b_avg$!
fwd_byts_b_avg˙˙˙˙˙˙˙˙˙
6
fwd_header_len$!
fwd_header_len˙˙˙˙˙˙˙˙˙
0
fwd_iat_max!
fwd_iat_max˙˙˙˙˙˙˙˙˙
2
fwd_iat_mean"
fwd_iat_mean˙˙˙˙˙˙˙˙˙
0
fwd_iat_min!
fwd_iat_min˙˙˙˙˙˙˙˙˙
0
fwd_iat_std!
fwd_iat_std˙˙˙˙˙˙˙˙˙
0
fwd_iat_tot!
fwd_iat_tot˙˙˙˙˙˙˙˙˙
8
fwd_pkt_len_max%"
fwd_pkt_len_max˙˙˙˙˙˙˙˙˙
:
fwd_pkt_len_mean&#
fwd_pkt_len_mean˙˙˙˙˙˙˙˙˙
8
fwd_pkt_len_min%"
fwd_pkt_len_min˙˙˙˙˙˙˙˙˙
8
fwd_pkt_len_std%"
fwd_pkt_len_std˙˙˙˙˙˙˙˙˙
6
fwd_pkts_b_avg$!
fwd_pkts_b_avg˙˙˙˙˙˙˙˙˙
.

fwd_pkts_s 

fwd_pkts_s˙˙˙˙˙˙˙˙˙
4
fwd_psh_flags# 
fwd_psh_flags˙˙˙˙˙˙˙˙˙
:
fwd_seg_size_min&#
fwd_seg_size_min˙˙˙˙˙˙˙˙˙
4
fwd_urg_flags# 
fwd_urg_flags˙˙˙˙˙˙˙˙˙
*
idle_max
idle_max˙˙˙˙˙˙˙˙˙
,
	idle_mean
	idle_mean˙˙˙˙˙˙˙˙˙
*
idle_min
idle_min˙˙˙˙˙˙˙˙˙
*
idle_std
idle_std˙˙˙˙˙˙˙˙˙
"
info
info˙˙˙˙˙˙˙˙˙
<
init_bwd_win_byts'$
init_bwd_win_byts˙˙˙˙˙˙˙˙˙
<
init_fwd_win_byts'$
init_fwd_win_byts˙˙˙˙˙˙˙˙˙
0
pkt_len_max!
pkt_len_max˙˙˙˙˙˙˙˙˙
2
pkt_len_mean"
pkt_len_mean˙˙˙˙˙˙˙˙˙
0
pkt_len_min!
pkt_len_min˙˙˙˙˙˙˙˙˙
0
pkt_len_std!
pkt_len_std˙˙˙˙˙˙˙˙˙
0
pkt_len_var!
pkt_len_var˙˙˙˙˙˙˙˙˙
.

pkt_length 

pkt_length˙˙˙˙˙˙˙˙˙
2
pkt_size_avg"
pkt_size_avg˙˙˙˙˙˙˙˙˙
*
protocol
protocol˙˙˙˙˙˙˙˙˙
2
psh_flag_cnt"
psh_flag_cnt˙˙˙˙˙˙˙˙˙
2
rst_flag_cnt"
rst_flag_cnt˙˙˙˙˙˙˙˙˙
&
src_ip
src_ip˙˙˙˙˙˙˙˙˙	
*
src_port
src_port˙˙˙˙˙˙˙˙˙
2
syn_flag_cnt"
syn_flag_cnt˙˙˙˙˙˙˙˙˙
,
	timestamp
	timestamp˙˙˙˙˙˙˙˙˙
2
tot_bwd_pkts"
tot_bwd_pkts˙˙˙˙˙˙˙˙˙
2
tot_fwd_pkts"
tot_fwd_pkts˙˙˙˙˙˙˙˙˙
8
totlen_bwd_pkts%"
totlen_bwd_pkts˙˙˙˙˙˙˙˙˙
8
totlen_fwd_pkts%"
totlen_fwd_pkts˙˙˙˙˙˙˙˙˙
2
urg_flag_cnt"
urg_flag_cnt˙˙˙˙˙˙˙˙˙"3Ş0
.
output_1"
output_1˙˙˙˙˙˙˙˙˙X
,__inference_yggdrasil_model_path_tensor_4165(.˘
˘
` 
Ş "
unknown 