╧Г
Е╒
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
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
$
DisableCopyOnRead
resourceИ
.
Identity

input"T
output"T"	
Ttype
Э
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	"
grad_abool( "
grad_bbool( 
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
0
Sigmoid
x"T
y"T"
Ttype:

2
┴
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
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
░
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
9
VarIsInitializedOp
resource
is_initialized
И"serve*2.17.02v2.17.0-rc1-2-gad6d8cc177d8▀·
б
Network/dense/biasVarHandleOp*
_output_shapes
: *#

debug_nameNetwork/dense/bias/*
dtype0*
shape:*#
shared_nameNetwork/dense/bias
u
&Network/dense/bias/Read/ReadVariableOpReadVariableOpNetwork/dense/bias*
_output_shapes
:*
dtype0
л
Network/dense/kernelVarHandleOp*
_output_shapes
: *%

debug_nameNetwork/dense/kernel/*
dtype0*
shape
:J*%
shared_nameNetwork/dense/kernel
}
(Network/dense/kernel/Read/ReadVariableOpReadVariableOpNetwork/dense/kernel*
_output_shapes

:J*
dtype0
з
Network/dense/bias_1VarHandleOp*
_output_shapes
: *%

debug_nameNetwork/dense/bias_1/*
dtype0*
shape:*%
shared_nameNetwork/dense/bias_1
y
(Network/dense/bias_1/Read/ReadVariableOpReadVariableOpNetwork/dense/bias_1*
_output_shapes
:*
dtype0
С
#Variable/Initializer/ReadVariableOpReadVariableOpNetwork/dense/bias_1*
_class
loc:@Variable*
_output_shapes
:*
dtype0
а
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape:*
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
_
Variable/AssignAssignVariableOpVariable#Variable/Initializer/ReadVariableOp*
dtype0
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0
▒
Network/dense/kernel_1VarHandleOp*
_output_shapes
: *'

debug_nameNetwork/dense/kernel_1/*
dtype0*
shape
:J*'
shared_nameNetwork/dense/kernel_1
Б
*Network/dense/kernel_1/Read/ReadVariableOpReadVariableOpNetwork/dense/kernel_1*
_output_shapes

:J*
dtype0
Ы
%Variable_1/Initializer/ReadVariableOpReadVariableOpNetwork/dense/kernel_1*
_class
loc:@Variable_1*
_output_shapes

:J*
dtype0
м

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape
:J*
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
e
Variable_1/AssignAssignVariableOp
Variable_1%Variable_1/Initializer/ReadVariableOp*
dtype0
i
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes

:J*
dtype0
u
serve_ack_flag_cntPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
s
serve_active_maxPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
t
serve_active_meanPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
s
serve_active_minPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
s
serve_active_stdPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
y
serve_bwd_blk_rate_avgPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
w
serve_bwd_byts_b_avgPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
w
serve_bwd_header_lenPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
t
serve_bwd_iat_maxPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
u
serve_bwd_iat_meanPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
t
serve_bwd_iat_minPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
t
serve_bwd_iat_stdPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
t
serve_bwd_iat_totPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
x
serve_bwd_pkt_len_maxPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
y
serve_bwd_pkt_len_meanPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
x
serve_bwd_pkt_len_minPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
x
serve_bwd_pkt_len_stdPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
w
serve_bwd_pkts_b_avgPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
s
serve_bwd_pkts_sPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
v
serve_bwd_psh_flagsPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
v
serve_bwd_urg_flagsPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
v
serve_down_up_ratioPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
q
serve_dst_portPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
u
serve_ece_flag_cntPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
u
serve_fin_flag_cntPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
t
serve_flow_byts_sPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
v
serve_flow_durationPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
u
serve_flow_iat_maxPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
v
serve_flow_iat_meanPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
u
serve_flow_iat_minPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
u
serve_flow_iat_stdPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
t
serve_flow_pkts_sPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
z
serve_fwd_act_data_pktsPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
y
serve_fwd_blk_rate_avgPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
w
serve_fwd_byts_b_avgPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
w
serve_fwd_header_lenPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
t
serve_fwd_iat_maxPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
u
serve_fwd_iat_meanPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
t
serve_fwd_iat_minPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
t
serve_fwd_iat_stdPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
t
serve_fwd_iat_totPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
x
serve_fwd_pkt_len_maxPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
y
serve_fwd_pkt_len_meanPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
x
serve_fwd_pkt_len_minPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
x
serve_fwd_pkt_len_stdPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
w
serve_fwd_pkts_b_avgPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
s
serve_fwd_pkts_sPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
v
serve_fwd_psh_flagsPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
y
serve_fwd_seg_size_minPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
v
serve_fwd_urg_flagsPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
q
serve_idle_maxPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
r
serve_idle_meanPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
q
serve_idle_minPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
q
serve_idle_stdPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
m

serve_infoPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
z
serve_init_bwd_win_bytsPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
z
serve_init_fwd_win_bytsPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
t
serve_pkt_len_maxPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
u
serve_pkt_len_meanPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
t
serve_pkt_len_minPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
t
serve_pkt_len_stdPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
t
serve_pkt_len_varPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
s
serve_pkt_lengthPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
u
serve_pkt_size_avgPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
q
serve_protocolPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
u
serve_psh_flag_cntPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
u
serve_rst_flag_cntPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
q
serve_src_portPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
u
serve_syn_flag_cntPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
u
serve_tot_bwd_pktsPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
u
serve_tot_fwd_pktsPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
x
serve_totlen_bwd_pktsPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
x
serve_totlen_fwd_pktsPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
u
serve_urg_flag_cntPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
Ї
StatefulPartitionedCallStatefulPartitionedCallserve_ack_flag_cntserve_active_maxserve_active_meanserve_active_minserve_active_stdserve_bwd_blk_rate_avgserve_bwd_byts_b_avgserve_bwd_header_lenserve_bwd_iat_maxserve_bwd_iat_meanserve_bwd_iat_minserve_bwd_iat_stdserve_bwd_iat_totserve_bwd_pkt_len_maxserve_bwd_pkt_len_meanserve_bwd_pkt_len_minserve_bwd_pkt_len_stdserve_bwd_pkts_b_avgserve_bwd_pkts_sserve_bwd_psh_flagsserve_bwd_urg_flagsserve_down_up_ratioserve_dst_portserve_ece_flag_cntserve_fin_flag_cntserve_flow_byts_sserve_flow_durationserve_flow_iat_maxserve_flow_iat_meanserve_flow_iat_minserve_flow_iat_stdserve_flow_pkts_sserve_fwd_act_data_pktsserve_fwd_blk_rate_avgserve_fwd_byts_b_avgserve_fwd_header_lenserve_fwd_iat_maxserve_fwd_iat_meanserve_fwd_iat_minserve_fwd_iat_stdserve_fwd_iat_totserve_fwd_pkt_len_maxserve_fwd_pkt_len_meanserve_fwd_pkt_len_minserve_fwd_pkt_len_stdserve_fwd_pkts_b_avgserve_fwd_pkts_sserve_fwd_psh_flagsserve_fwd_seg_size_minserve_fwd_urg_flagsserve_idle_maxserve_idle_meanserve_idle_minserve_idle_std
serve_infoserve_init_bwd_win_bytsserve_init_fwd_win_bytsserve_pkt_len_maxserve_pkt_len_meanserve_pkt_len_minserve_pkt_len_stdserve_pkt_len_varserve_pkt_lengthserve_pkt_size_avgserve_protocolserve_psh_flag_cntserve_rst_flag_cntserve_src_portserve_syn_flag_cntserve_tot_bwd_pktsserve_tot_fwd_pktsserve_totlen_bwd_pktsserve_totlen_fwd_pktsserve_urg_flag_cntNetwork/dense/kernel_1Network/dense/bias_1*W
TinP
N2L*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
JK*-
config_proto

CPU

GPU 2J 8В *5
f0R.
,__inference_signature_wrapper___call___17769

serving_default_ack_flag_cntPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
}
serving_default_active_maxPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
~
serving_default_active_meanPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
}
serving_default_active_minPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
}
serving_default_active_stdPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
Г
 serving_default_bwd_blk_rate_avgPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
Б
serving_default_bwd_byts_b_avgPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
Б
serving_default_bwd_header_lenPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
~
serving_default_bwd_iat_maxPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         

serving_default_bwd_iat_meanPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
~
serving_default_bwd_iat_minPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
~
serving_default_bwd_iat_stdPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
~
serving_default_bwd_iat_totPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
В
serving_default_bwd_pkt_len_maxPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
Г
 serving_default_bwd_pkt_len_meanPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
В
serving_default_bwd_pkt_len_minPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
В
serving_default_bwd_pkt_len_stdPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
Б
serving_default_bwd_pkts_b_avgPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
}
serving_default_bwd_pkts_sPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
А
serving_default_bwd_psh_flagsPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
А
serving_default_bwd_urg_flagsPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
А
serving_default_down_up_ratioPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
{
serving_default_dst_portPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         

serving_default_ece_flag_cntPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         

serving_default_fin_flag_cntPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
~
serving_default_flow_byts_sPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
А
serving_default_flow_durationPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         

serving_default_flow_iat_maxPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
А
serving_default_flow_iat_meanPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         

serving_default_flow_iat_minPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         

serving_default_flow_iat_stdPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
~
serving_default_flow_pkts_sPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
Д
!serving_default_fwd_act_data_pktsPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
Г
 serving_default_fwd_blk_rate_avgPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
Б
serving_default_fwd_byts_b_avgPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
Б
serving_default_fwd_header_lenPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
~
serving_default_fwd_iat_maxPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         

serving_default_fwd_iat_meanPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
~
serving_default_fwd_iat_minPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
~
serving_default_fwd_iat_stdPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
~
serving_default_fwd_iat_totPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
В
serving_default_fwd_pkt_len_maxPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
Г
 serving_default_fwd_pkt_len_meanPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
В
serving_default_fwd_pkt_len_minPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
В
serving_default_fwd_pkt_len_stdPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
Б
serving_default_fwd_pkts_b_avgPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
}
serving_default_fwd_pkts_sPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
А
serving_default_fwd_psh_flagsPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
Г
 serving_default_fwd_seg_size_minPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
А
serving_default_fwd_urg_flagsPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
{
serving_default_idle_maxPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
|
serving_default_idle_meanPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
{
serving_default_idle_minPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
{
serving_default_idle_stdPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
w
serving_default_infoPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
Д
!serving_default_init_bwd_win_bytsPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
Д
!serving_default_init_fwd_win_bytsPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
~
serving_default_pkt_len_maxPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         

serving_default_pkt_len_meanPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
~
serving_default_pkt_len_minPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
~
serving_default_pkt_len_stdPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
~
serving_default_pkt_len_varPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
}
serving_default_pkt_lengthPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         

serving_default_pkt_size_avgPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
{
serving_default_protocolPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         

serving_default_psh_flag_cntPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         

serving_default_rst_flag_cntPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
{
serving_default_src_portPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         

serving_default_syn_flag_cntPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         

serving_default_tot_bwd_pktsPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         

serving_default_tot_fwd_pktsPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
В
serving_default_totlen_bwd_pktsPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
В
serving_default_totlen_fwd_pktsPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         

serving_default_urg_flag_cntPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
┌
StatefulPartitionedCall_1StatefulPartitionedCallserving_default_ack_flag_cntserving_default_active_maxserving_default_active_meanserving_default_active_minserving_default_active_std serving_default_bwd_blk_rate_avgserving_default_bwd_byts_b_avgserving_default_bwd_header_lenserving_default_bwd_iat_maxserving_default_bwd_iat_meanserving_default_bwd_iat_minserving_default_bwd_iat_stdserving_default_bwd_iat_totserving_default_bwd_pkt_len_max serving_default_bwd_pkt_len_meanserving_default_bwd_pkt_len_minserving_default_bwd_pkt_len_stdserving_default_bwd_pkts_b_avgserving_default_bwd_pkts_sserving_default_bwd_psh_flagsserving_default_bwd_urg_flagsserving_default_down_up_ratioserving_default_dst_portserving_default_ece_flag_cntserving_default_fin_flag_cntserving_default_flow_byts_sserving_default_flow_durationserving_default_flow_iat_maxserving_default_flow_iat_meanserving_default_flow_iat_minserving_default_flow_iat_stdserving_default_flow_pkts_s!serving_default_fwd_act_data_pkts serving_default_fwd_blk_rate_avgserving_default_fwd_byts_b_avgserving_default_fwd_header_lenserving_default_fwd_iat_maxserving_default_fwd_iat_meanserving_default_fwd_iat_minserving_default_fwd_iat_stdserving_default_fwd_iat_totserving_default_fwd_pkt_len_max serving_default_fwd_pkt_len_meanserving_default_fwd_pkt_len_minserving_default_fwd_pkt_len_stdserving_default_fwd_pkts_b_avgserving_default_fwd_pkts_sserving_default_fwd_psh_flags serving_default_fwd_seg_size_minserving_default_fwd_urg_flagsserving_default_idle_maxserving_default_idle_meanserving_default_idle_minserving_default_idle_stdserving_default_info!serving_default_init_bwd_win_byts!serving_default_init_fwd_win_bytsserving_default_pkt_len_maxserving_default_pkt_len_meanserving_default_pkt_len_minserving_default_pkt_len_stdserving_default_pkt_len_varserving_default_pkt_lengthserving_default_pkt_size_avgserving_default_protocolserving_default_psh_flag_cntserving_default_rst_flag_cntserving_default_src_portserving_default_syn_flag_cntserving_default_tot_bwd_pktsserving_default_tot_fwd_pktsserving_default_totlen_bwd_pktsserving_default_totlen_fwd_pktsserving_default_urg_flag_cntNetwork/dense/kernel_1Network/dense/bias_1*W
TinP
N2L*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
JK*-
config_proto

CPU

GPU 2J 8В *5
f0R.
,__inference_signature_wrapper___call___17851

NoOpNoOp
╣
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ї
valueъBч Bр
К
	variables
trainable_variables
non_trainable_variables
_all_variables
_misc_assets
	serve

signatures*

0
	1*

0
	1*
* 


0
1*
* 

trace_0* 
"
	serve
serving_default* 
JD
VARIABLE_VALUE
Variable_1&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
HB
VARIABLE_VALUEVariable&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUENetwork/dense/kernel_1+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUENetwork/dense/bias_1+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
т
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename
Variable_1VariableNetwork/dense/kernel_1Network/dense/bias_1Const*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *'
f"R 
__inference__traced_save_18053
▌
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filename
Variable_1VariableNetwork/dense/kernel_1Network/dense/bias_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__traced_restore_18074ки
нW
╤
,__inference_signature_wrapper___call___17769
ack_flag_cnt

active_max
active_mean

active_min

active_std
bwd_blk_rate_avg
bwd_byts_b_avg
bwd_header_len
bwd_iat_max
bwd_iat_mean
bwd_iat_min
bwd_iat_std
bwd_iat_tot
bwd_pkt_len_max
bwd_pkt_len_mean
bwd_pkt_len_min
bwd_pkt_len_std
bwd_pkts_b_avg

bwd_pkts_s
bwd_psh_flags
bwd_urg_flags
down_up_ratio
dst_port
ece_flag_cnt
fin_flag_cnt
flow_byts_s
flow_duration
flow_iat_max
flow_iat_mean
flow_iat_min
flow_iat_std
flow_pkts_s
fwd_act_data_pkts
fwd_blk_rate_avg
fwd_byts_b_avg
fwd_header_len
fwd_iat_max
fwd_iat_mean
fwd_iat_min
fwd_iat_std
fwd_iat_tot
fwd_pkt_len_max
fwd_pkt_len_mean
fwd_pkt_len_min
fwd_pkt_len_std
fwd_pkts_b_avg

fwd_pkts_s
fwd_psh_flags
fwd_seg_size_min
fwd_urg_flags
idle_max
	idle_mean
idle_min
idle_std
info
init_bwd_win_byts
init_fwd_win_byts
pkt_len_max
pkt_len_mean
pkt_len_min
pkt_len_std
pkt_len_var

pkt_length
pkt_size_avg
protocol
psh_flag_cnt
rst_flag_cnt
src_port
syn_flag_cnt
tot_bwd_pkts
tot_fwd_pkts
totlen_bwd_pkts
totlen_fwd_pkts
urg_flag_cnt
unknown:J
	unknown_0:
identityИвStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallack_flag_cnt
active_maxactive_mean
active_min
active_stdbwd_blk_rate_avgbwd_byts_b_avgbwd_header_lenbwd_iat_maxbwd_iat_meanbwd_iat_minbwd_iat_stdbwd_iat_totbwd_pkt_len_maxbwd_pkt_len_meanbwd_pkt_len_minbwd_pkt_len_stdbwd_pkts_b_avg
bwd_pkts_sbwd_psh_flagsbwd_urg_flagsdown_up_ratiodst_portece_flag_cntfin_flag_cntflow_byts_sflow_durationflow_iat_maxflow_iat_meanflow_iat_minflow_iat_stdflow_pkts_sfwd_act_data_pktsfwd_blk_rate_avgfwd_byts_b_avgfwd_header_lenfwd_iat_maxfwd_iat_meanfwd_iat_minfwd_iat_stdfwd_iat_totfwd_pkt_len_maxfwd_pkt_len_meanfwd_pkt_len_minfwd_pkt_len_stdfwd_pkts_b_avg
fwd_pkts_sfwd_psh_flagsfwd_seg_size_minfwd_urg_flagsidle_max	idle_meanidle_minidle_stdinfoinit_bwd_win_bytsinit_fwd_win_bytspkt_len_maxpkt_len_meanpkt_len_minpkt_len_stdpkt_len_var
pkt_lengthpkt_size_avgprotocolpsh_flag_cntrst_flag_cntsrc_portsyn_flag_cnttot_bwd_pktstot_fwd_pktstotlen_bwd_pktstotlen_fwd_pktsurg_flag_cntunknown	unknown_0*W
TinP
N2L*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
JK*-
config_proto

CPU

GPU 2J 8В *#
fR
__inference___call___17686o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ч
_input_shapesЕ
В:         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         : : 22
StatefulPartitionedCallStatefulPartitionedCall:%K!

_user_specified_name17765:%J!

_user_specified_name17763:UIQ
'
_output_shapes
:         
&
_user_specified_nameurg_flag_cnt:XHT
'
_output_shapes
:         
)
_user_specified_nametotlen_fwd_pkts:XGT
'
_output_shapes
:         
)
_user_specified_nametotlen_bwd_pkts:UFQ
'
_output_shapes
:         
&
_user_specified_nametot_fwd_pkts:UEQ
'
_output_shapes
:         
&
_user_specified_nametot_bwd_pkts:UDQ
'
_output_shapes
:         
&
_user_specified_namesyn_flag_cnt:QCM
'
_output_shapes
:         
"
_user_specified_name
src_port:UBQ
'
_output_shapes
:         
&
_user_specified_namerst_flag_cnt:UAQ
'
_output_shapes
:         
&
_user_specified_namepsh_flag_cnt:Q@M
'
_output_shapes
:         
"
_user_specified_name
protocol:U?Q
'
_output_shapes
:         
&
_user_specified_namepkt_size_avg:S>O
'
_output_shapes
:         
$
_user_specified_name
pkt_length:T=P
'
_output_shapes
:         
%
_user_specified_namepkt_len_var:T<P
'
_output_shapes
:         
%
_user_specified_namepkt_len_std:T;P
'
_output_shapes
:         
%
_user_specified_namepkt_len_min:U:Q
'
_output_shapes
:         
&
_user_specified_namepkt_len_mean:T9P
'
_output_shapes
:         
%
_user_specified_namepkt_len_max:Z8V
'
_output_shapes
:         
+
_user_specified_nameinit_fwd_win_byts:Z7V
'
_output_shapes
:         
+
_user_specified_nameinit_bwd_win_byts:M6I
'
_output_shapes
:         

_user_specified_nameinfo:Q5M
'
_output_shapes
:         
"
_user_specified_name
idle_std:Q4M
'
_output_shapes
:         
"
_user_specified_name
idle_min:R3N
'
_output_shapes
:         
#
_user_specified_name	idle_mean:Q2M
'
_output_shapes
:         
"
_user_specified_name
idle_max:V1R
'
_output_shapes
:         
'
_user_specified_namefwd_urg_flags:Y0U
'
_output_shapes
:         
*
_user_specified_namefwd_seg_size_min:V/R
'
_output_shapes
:         
'
_user_specified_namefwd_psh_flags:S.O
'
_output_shapes
:         
$
_user_specified_name
fwd_pkts_s:W-S
'
_output_shapes
:         
(
_user_specified_namefwd_pkts_b_avg:X,T
'
_output_shapes
:         
)
_user_specified_namefwd_pkt_len_std:X+T
'
_output_shapes
:         
)
_user_specified_namefwd_pkt_len_min:Y*U
'
_output_shapes
:         
*
_user_specified_namefwd_pkt_len_mean:X)T
'
_output_shapes
:         
)
_user_specified_namefwd_pkt_len_max:T(P
'
_output_shapes
:         
%
_user_specified_namefwd_iat_tot:T'P
'
_output_shapes
:         
%
_user_specified_namefwd_iat_std:T&P
'
_output_shapes
:         
%
_user_specified_namefwd_iat_min:U%Q
'
_output_shapes
:         
&
_user_specified_namefwd_iat_mean:T$P
'
_output_shapes
:         
%
_user_specified_namefwd_iat_max:W#S
'
_output_shapes
:         
(
_user_specified_namefwd_header_len:W"S
'
_output_shapes
:         
(
_user_specified_namefwd_byts_b_avg:Y!U
'
_output_shapes
:         
*
_user_specified_namefwd_blk_rate_avg:Z V
'
_output_shapes
:         
+
_user_specified_namefwd_act_data_pkts:TP
'
_output_shapes
:         
%
_user_specified_nameflow_pkts_s:UQ
'
_output_shapes
:         
&
_user_specified_nameflow_iat_std:UQ
'
_output_shapes
:         
&
_user_specified_nameflow_iat_min:VR
'
_output_shapes
:         
'
_user_specified_nameflow_iat_mean:UQ
'
_output_shapes
:         
&
_user_specified_nameflow_iat_max:VR
'
_output_shapes
:         
'
_user_specified_nameflow_duration:TP
'
_output_shapes
:         
%
_user_specified_nameflow_byts_s:UQ
'
_output_shapes
:         
&
_user_specified_namefin_flag_cnt:UQ
'
_output_shapes
:         
&
_user_specified_nameece_flag_cnt:QM
'
_output_shapes
:         
"
_user_specified_name
dst_port:VR
'
_output_shapes
:         
'
_user_specified_namedown_up_ratio:VR
'
_output_shapes
:         
'
_user_specified_namebwd_urg_flags:VR
'
_output_shapes
:         
'
_user_specified_namebwd_psh_flags:SO
'
_output_shapes
:         
$
_user_specified_name
bwd_pkts_s:WS
'
_output_shapes
:         
(
_user_specified_namebwd_pkts_b_avg:XT
'
_output_shapes
:         
)
_user_specified_namebwd_pkt_len_std:XT
'
_output_shapes
:         
)
_user_specified_namebwd_pkt_len_min:YU
'
_output_shapes
:         
*
_user_specified_namebwd_pkt_len_mean:XT
'
_output_shapes
:         
)
_user_specified_namebwd_pkt_len_max:TP
'
_output_shapes
:         
%
_user_specified_namebwd_iat_tot:TP
'
_output_shapes
:         
%
_user_specified_namebwd_iat_std:T
P
'
_output_shapes
:         
%
_user_specified_namebwd_iat_min:U	Q
'
_output_shapes
:         
&
_user_specified_namebwd_iat_mean:TP
'
_output_shapes
:         
%
_user_specified_namebwd_iat_max:WS
'
_output_shapes
:         
(
_user_specified_namebwd_header_len:WS
'
_output_shapes
:         
(
_user_specified_namebwd_byts_b_avg:YU
'
_output_shapes
:         
*
_user_specified_namebwd_blk_rate_avg:SO
'
_output_shapes
:         
$
_user_specified_name
active_std:SO
'
_output_shapes
:         
$
_user_specified_name
active_min:TP
'
_output_shapes
:         
%
_user_specified_nameactive_mean:SO
'
_output_shapes
:         
$
_user_specified_name
active_max:U Q
'
_output_shapes
:         
&
_user_specified_nameack_flag_cnt
нW
╤
,__inference_signature_wrapper___call___17851
ack_flag_cnt

active_max
active_mean

active_min

active_std
bwd_blk_rate_avg
bwd_byts_b_avg
bwd_header_len
bwd_iat_max
bwd_iat_mean
bwd_iat_min
bwd_iat_std
bwd_iat_tot
bwd_pkt_len_max
bwd_pkt_len_mean
bwd_pkt_len_min
bwd_pkt_len_std
bwd_pkts_b_avg

bwd_pkts_s
bwd_psh_flags
bwd_urg_flags
down_up_ratio
dst_port
ece_flag_cnt
fin_flag_cnt
flow_byts_s
flow_duration
flow_iat_max
flow_iat_mean
flow_iat_min
flow_iat_std
flow_pkts_s
fwd_act_data_pkts
fwd_blk_rate_avg
fwd_byts_b_avg
fwd_header_len
fwd_iat_max
fwd_iat_mean
fwd_iat_min
fwd_iat_std
fwd_iat_tot
fwd_pkt_len_max
fwd_pkt_len_mean
fwd_pkt_len_min
fwd_pkt_len_std
fwd_pkts_b_avg

fwd_pkts_s
fwd_psh_flags
fwd_seg_size_min
fwd_urg_flags
idle_max
	idle_mean
idle_min
idle_std
info
init_bwd_win_byts
init_fwd_win_byts
pkt_len_max
pkt_len_mean
pkt_len_min
pkt_len_std
pkt_len_var

pkt_length
pkt_size_avg
protocol
psh_flag_cnt
rst_flag_cnt
src_port
syn_flag_cnt
tot_bwd_pkts
tot_fwd_pkts
totlen_bwd_pkts
totlen_fwd_pkts
urg_flag_cnt
unknown:J
	unknown_0:
identityИвStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallack_flag_cnt
active_maxactive_mean
active_min
active_stdbwd_blk_rate_avgbwd_byts_b_avgbwd_header_lenbwd_iat_maxbwd_iat_meanbwd_iat_minbwd_iat_stdbwd_iat_totbwd_pkt_len_maxbwd_pkt_len_meanbwd_pkt_len_minbwd_pkt_len_stdbwd_pkts_b_avg
bwd_pkts_sbwd_psh_flagsbwd_urg_flagsdown_up_ratiodst_portece_flag_cntfin_flag_cntflow_byts_sflow_durationflow_iat_maxflow_iat_meanflow_iat_minflow_iat_stdflow_pkts_sfwd_act_data_pktsfwd_blk_rate_avgfwd_byts_b_avgfwd_header_lenfwd_iat_maxfwd_iat_meanfwd_iat_minfwd_iat_stdfwd_iat_totfwd_pkt_len_maxfwd_pkt_len_meanfwd_pkt_len_minfwd_pkt_len_stdfwd_pkts_b_avg
fwd_pkts_sfwd_psh_flagsfwd_seg_size_minfwd_urg_flagsidle_max	idle_meanidle_minidle_stdinfoinit_bwd_win_bytsinit_fwd_win_bytspkt_len_maxpkt_len_meanpkt_len_minpkt_len_stdpkt_len_var
pkt_lengthpkt_size_avgprotocolpsh_flag_cntrst_flag_cntsrc_portsyn_flag_cnttot_bwd_pktstot_fwd_pktstotlen_bwd_pktstotlen_fwd_pktsurg_flag_cntunknown	unknown_0*W
TinP
N2L*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
JK*-
config_proto

CPU

GPU 2J 8В *#
fR
__inference___call___17686o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ч
_input_shapesЕ
В:         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         : : 22
StatefulPartitionedCallStatefulPartitionedCall:%K!

_user_specified_name17847:%J!

_user_specified_name17845:UIQ
'
_output_shapes
:         
&
_user_specified_nameurg_flag_cnt:XHT
'
_output_shapes
:         
)
_user_specified_nametotlen_fwd_pkts:XGT
'
_output_shapes
:         
)
_user_specified_nametotlen_bwd_pkts:UFQ
'
_output_shapes
:         
&
_user_specified_nametot_fwd_pkts:UEQ
'
_output_shapes
:         
&
_user_specified_nametot_bwd_pkts:UDQ
'
_output_shapes
:         
&
_user_specified_namesyn_flag_cnt:QCM
'
_output_shapes
:         
"
_user_specified_name
src_port:UBQ
'
_output_shapes
:         
&
_user_specified_namerst_flag_cnt:UAQ
'
_output_shapes
:         
&
_user_specified_namepsh_flag_cnt:Q@M
'
_output_shapes
:         
"
_user_specified_name
protocol:U?Q
'
_output_shapes
:         
&
_user_specified_namepkt_size_avg:S>O
'
_output_shapes
:         
$
_user_specified_name
pkt_length:T=P
'
_output_shapes
:         
%
_user_specified_namepkt_len_var:T<P
'
_output_shapes
:         
%
_user_specified_namepkt_len_std:T;P
'
_output_shapes
:         
%
_user_specified_namepkt_len_min:U:Q
'
_output_shapes
:         
&
_user_specified_namepkt_len_mean:T9P
'
_output_shapes
:         
%
_user_specified_namepkt_len_max:Z8V
'
_output_shapes
:         
+
_user_specified_nameinit_fwd_win_byts:Z7V
'
_output_shapes
:         
+
_user_specified_nameinit_bwd_win_byts:M6I
'
_output_shapes
:         

_user_specified_nameinfo:Q5M
'
_output_shapes
:         
"
_user_specified_name
idle_std:Q4M
'
_output_shapes
:         
"
_user_specified_name
idle_min:R3N
'
_output_shapes
:         
#
_user_specified_name	idle_mean:Q2M
'
_output_shapes
:         
"
_user_specified_name
idle_max:V1R
'
_output_shapes
:         
'
_user_specified_namefwd_urg_flags:Y0U
'
_output_shapes
:         
*
_user_specified_namefwd_seg_size_min:V/R
'
_output_shapes
:         
'
_user_specified_namefwd_psh_flags:S.O
'
_output_shapes
:         
$
_user_specified_name
fwd_pkts_s:W-S
'
_output_shapes
:         
(
_user_specified_namefwd_pkts_b_avg:X,T
'
_output_shapes
:         
)
_user_specified_namefwd_pkt_len_std:X+T
'
_output_shapes
:         
)
_user_specified_namefwd_pkt_len_min:Y*U
'
_output_shapes
:         
*
_user_specified_namefwd_pkt_len_mean:X)T
'
_output_shapes
:         
)
_user_specified_namefwd_pkt_len_max:T(P
'
_output_shapes
:         
%
_user_specified_namefwd_iat_tot:T'P
'
_output_shapes
:         
%
_user_specified_namefwd_iat_std:T&P
'
_output_shapes
:         
%
_user_specified_namefwd_iat_min:U%Q
'
_output_shapes
:         
&
_user_specified_namefwd_iat_mean:T$P
'
_output_shapes
:         
%
_user_specified_namefwd_iat_max:W#S
'
_output_shapes
:         
(
_user_specified_namefwd_header_len:W"S
'
_output_shapes
:         
(
_user_specified_namefwd_byts_b_avg:Y!U
'
_output_shapes
:         
*
_user_specified_namefwd_blk_rate_avg:Z V
'
_output_shapes
:         
+
_user_specified_namefwd_act_data_pkts:TP
'
_output_shapes
:         
%
_user_specified_nameflow_pkts_s:UQ
'
_output_shapes
:         
&
_user_specified_nameflow_iat_std:UQ
'
_output_shapes
:         
&
_user_specified_nameflow_iat_min:VR
'
_output_shapes
:         
'
_user_specified_nameflow_iat_mean:UQ
'
_output_shapes
:         
&
_user_specified_nameflow_iat_max:VR
'
_output_shapes
:         
'
_user_specified_nameflow_duration:TP
'
_output_shapes
:         
%
_user_specified_nameflow_byts_s:UQ
'
_output_shapes
:         
&
_user_specified_namefin_flag_cnt:UQ
'
_output_shapes
:         
&
_user_specified_nameece_flag_cnt:QM
'
_output_shapes
:         
"
_user_specified_name
dst_port:VR
'
_output_shapes
:         
'
_user_specified_namedown_up_ratio:VR
'
_output_shapes
:         
'
_user_specified_namebwd_urg_flags:VR
'
_output_shapes
:         
'
_user_specified_namebwd_psh_flags:SO
'
_output_shapes
:         
$
_user_specified_name
bwd_pkts_s:WS
'
_output_shapes
:         
(
_user_specified_namebwd_pkts_b_avg:XT
'
_output_shapes
:         
)
_user_specified_namebwd_pkt_len_std:XT
'
_output_shapes
:         
)
_user_specified_namebwd_pkt_len_min:YU
'
_output_shapes
:         
*
_user_specified_namebwd_pkt_len_mean:XT
'
_output_shapes
:         
)
_user_specified_namebwd_pkt_len_max:TP
'
_output_shapes
:         
%
_user_specified_namebwd_iat_tot:TP
'
_output_shapes
:         
%
_user_specified_namebwd_iat_std:T
P
'
_output_shapes
:         
%
_user_specified_namebwd_iat_min:U	Q
'
_output_shapes
:         
&
_user_specified_namebwd_iat_mean:TP
'
_output_shapes
:         
%
_user_specified_namebwd_iat_max:WS
'
_output_shapes
:         
(
_user_specified_namebwd_header_len:WS
'
_output_shapes
:         
(
_user_specified_namebwd_byts_b_avg:YU
'
_output_shapes
:         
*
_user_specified_namebwd_blk_rate_avg:SO
'
_output_shapes
:         
$
_user_specified_name
active_std:SO
'
_output_shapes
:         
$
_user_specified_name
active_min:TP
'
_output_shapes
:         
%
_user_specified_nameactive_mean:SO
'
_output_shapes
:         
$
_user_specified_name
active_max:U Q
'
_output_shapes
:         
&
_user_specified_nameack_flag_cnt
Б,
Щ
__inference__traced_save_18053
file_prefix3
!read_disablecopyonread_variable_1:J/
!read_1_disablecopyonread_variable:A
/read_2_disablecopyonread_network_dense_kernel_1:J;
-read_3_disablecopyonread_network_dense_bias_1:
savev2_const

identity_9ИвMergeV2CheckpointsвRead/DisableCopyOnReadвRead/ReadVariableOpвRead_1/DisableCopyOnReadвRead_1/ReadVariableOpвRead_2/DisableCopyOnReadвRead_2/ReadVariableOpвRead_3/DisableCopyOnReadвRead_3/ReadVariableOpw
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
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: d
Read/DisableCopyOnReadDisableCopyOnRead!read_disablecopyonread_variable_1*
_output_shapes
 О
Read/ReadVariableOpReadVariableOp!read_disablecopyonread_variable_1^Read/DisableCopyOnRead*
_output_shapes

:J*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:Ja

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:Jf
Read_1/DisableCopyOnReadDisableCopyOnRead!read_1_disablecopyonread_variable*
_output_shapes
 О
Read_1/ReadVariableOpReadVariableOp!read_1_disablecopyonread_variable^Read_1/DisableCopyOnRead*
_output_shapes
:*
dtype0Z

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_2/DisableCopyOnReadDisableCopyOnRead/read_2_disablecopyonread_network_dense_kernel_1*
_output_shapes
 а
Read_2/ReadVariableOpReadVariableOp/read_2_disablecopyonread_network_dense_kernel_1^Read_2/DisableCopyOnRead*
_output_shapes

:J*
dtype0^

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:Jc

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:Jr
Read_3/DisableCopyOnReadDisableCopyOnRead-read_3_disablecopyonread_network_dense_bias_1*
_output_shapes
 Ъ
Read_3/ReadVariableOpReadVariableOp-read_3_disablecopyonread_network_dense_bias_1^Read_3/DisableCopyOnRead*
_output_shapes
:*
dtype0Z

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ┤
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*▌
value╙B╨B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHw
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B ░
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes	
2Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 h

Identity_8Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: S

Identity_9IdentityIdentity_8:output:0^NoOp*
T0*
_output_shapes
:  
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp*
_output_shapes
 "!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:40
.
_user_specified_nameNetwork/dense/bias_1:62
0
_user_specified_nameNetwork/dense/kernel_1:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
П╡
У
__inference___call___17686
ack_flag_cnt

active_max
active_mean

active_min

active_std
bwd_blk_rate_avg
bwd_byts_b_avg
bwd_header_len
bwd_iat_max
bwd_iat_mean
bwd_iat_min
bwd_iat_std
bwd_iat_tot
bwd_pkt_len_max
bwd_pkt_len_mean
bwd_pkt_len_min
bwd_pkt_len_std
bwd_pkts_b_avg

bwd_pkts_s
bwd_psh_flags
bwd_urg_flags
down_up_ratio
dst_port
ece_flag_cnt
fin_flag_cnt
flow_byts_s
flow_duration
flow_iat_max
flow_iat_mean
flow_iat_min
flow_iat_std
flow_pkts_s
fwd_act_data_pkts
fwd_blk_rate_avg
fwd_byts_b_avg
fwd_header_len
fwd_iat_max
fwd_iat_mean
fwd_iat_min
fwd_iat_std
fwd_iat_tot
fwd_pkt_len_max
fwd_pkt_len_mean
fwd_pkt_len_min
fwd_pkt_len_std
fwd_pkts_b_avg

fwd_pkts_s
fwd_psh_flags
fwd_seg_size_min
fwd_urg_flags
idle_max
	idle_mean
idle_min
idle_std
info
init_bwd_win_byts
init_fwd_win_byts
pkt_len_max
pkt_len_mean
pkt_len_min
pkt_len_std
pkt_len_var

pkt_length
pkt_size_avg
protocol
psh_flag_cnt
rst_flag_cnt
src_port
syn_flag_cnt
tot_bwd_pkts
tot_fwd_pkts
totlen_bwd_pkts
totlen_fwd_pkts
urg_flag_cntU
Clogisticregression_1_network_1_dense_1_cast_readvariableop_resource:JP
Blogisticregression_1_network_1_dense_1_add_readvariableop_resource:
identityИв9LogisticRegression_1/Network_1/dense_1/Add/ReadVariableOpв:LogisticRegression_1/Network_1/dense_1/Cast/ReadVariableOph
LogisticRegression_1/CastCastinfo*

DstT0*

SrcT0*'
_output_shapes
:         s
LogisticRegression_1/Cast_1Castflow_duration*

DstT0*

SrcT0*'
_output_shapes
:         q
LogisticRegression_1/Cast_2Castflow_byts_s*

DstT0*

SrcT0*'
_output_shapes
:         q
LogisticRegression_1/Cast_3Castflow_pkts_s*

DstT0*

SrcT0*'
_output_shapes
:         p
LogisticRegression_1/Cast_4Cast
fwd_pkts_s*

DstT0*

SrcT0*'
_output_shapes
:         p
LogisticRegression_1/Cast_5Cast
bwd_pkts_s*

DstT0*

SrcT0*'
_output_shapes
:         r
LogisticRegression_1/Cast_6Casttot_fwd_pkts*

DstT0*

SrcT0*'
_output_shapes
:         r
LogisticRegression_1/Cast_7Casttot_bwd_pkts*

DstT0*

SrcT0*'
_output_shapes
:         u
LogisticRegression_1/Cast_8Casttotlen_fwd_pkts*

DstT0*

SrcT0*'
_output_shapes
:         u
LogisticRegression_1/Cast_9Casttotlen_bwd_pkts*

DstT0*

SrcT0*'
_output_shapes
:         v
LogisticRegression_1/Cast_10Castfwd_pkt_len_max*

DstT0*

SrcT0*'
_output_shapes
:         v
LogisticRegression_1/Cast_11Castfwd_pkt_len_min*

DstT0*

SrcT0*'
_output_shapes
:         w
LogisticRegression_1/Cast_12Castfwd_pkt_len_mean*

DstT0*

SrcT0*'
_output_shapes
:         v
LogisticRegression_1/Cast_13Castfwd_pkt_len_std*

DstT0*

SrcT0*'
_output_shapes
:         v
LogisticRegression_1/Cast_14Castbwd_pkt_len_max*

DstT0*

SrcT0*'
_output_shapes
:         v
LogisticRegression_1/Cast_15Castbwd_pkt_len_min*

DstT0*

SrcT0*'
_output_shapes
:         w
LogisticRegression_1/Cast_16Castbwd_pkt_len_mean*

DstT0*

SrcT0*'
_output_shapes
:         v
LogisticRegression_1/Cast_17Castbwd_pkt_len_std*

DstT0*

SrcT0*'
_output_shapes
:         r
LogisticRegression_1/Cast_18Castpkt_len_max*

DstT0*

SrcT0*'
_output_shapes
:         r
LogisticRegression_1/Cast_19Castpkt_len_min*

DstT0*

SrcT0*'
_output_shapes
:         s
LogisticRegression_1/Cast_20Castpkt_len_mean*

DstT0*

SrcT0*'
_output_shapes
:         r
LogisticRegression_1/Cast_21Castpkt_len_std*

DstT0*

SrcT0*'
_output_shapes
:         r
LogisticRegression_1/Cast_22Castpkt_len_var*

DstT0*

SrcT0*'
_output_shapes
:         w
LogisticRegression_1/Cast_23Castfwd_seg_size_min*

DstT0*

SrcT0*'
_output_shapes
:         x
LogisticRegression_1/Cast_24Castfwd_act_data_pkts*

DstT0*

SrcT0*'
_output_shapes
:         t
LogisticRegression_1/Cast_25Castflow_iat_mean*

DstT0*

SrcT0*'
_output_shapes
:         s
LogisticRegression_1/Cast_26Castflow_iat_max*

DstT0*

SrcT0*'
_output_shapes
:         s
LogisticRegression_1/Cast_27Castflow_iat_min*

DstT0*

SrcT0*'
_output_shapes
:         s
LogisticRegression_1/Cast_28Castflow_iat_std*

DstT0*

SrcT0*'
_output_shapes
:         r
LogisticRegression_1/Cast_29Castfwd_iat_tot*

DstT0*

SrcT0*'
_output_shapes
:         r
LogisticRegression_1/Cast_30Castfwd_iat_max*

DstT0*

SrcT0*'
_output_shapes
:         r
LogisticRegression_1/Cast_31Castfwd_iat_min*

DstT0*

SrcT0*'
_output_shapes
:         s
LogisticRegression_1/Cast_32Castfwd_iat_mean*

DstT0*

SrcT0*'
_output_shapes
:         r
LogisticRegression_1/Cast_33Castfwd_iat_std*

DstT0*

SrcT0*'
_output_shapes
:         r
LogisticRegression_1/Cast_34Castbwd_iat_tot*

DstT0*

SrcT0*'
_output_shapes
:         r
LogisticRegression_1/Cast_35Castbwd_iat_max*

DstT0*

SrcT0*'
_output_shapes
:         r
LogisticRegression_1/Cast_36Castbwd_iat_min*

DstT0*

SrcT0*'
_output_shapes
:         s
LogisticRegression_1/Cast_37Castbwd_iat_mean*

DstT0*

SrcT0*'
_output_shapes
:         r
LogisticRegression_1/Cast_38Castbwd_iat_std*

DstT0*

SrcT0*'
_output_shapes
:         t
LogisticRegression_1/Cast_39Castdown_up_ratio*

DstT0*

SrcT0*'
_output_shapes
:         s
LogisticRegression_1/Cast_40Castpkt_size_avg*

DstT0*

SrcT0*'
_output_shapes
:         x
LogisticRegression_1/Cast_41Castinit_fwd_win_byts*

DstT0*

SrcT0*'
_output_shapes
:         x
LogisticRegression_1/Cast_42Castinit_bwd_win_byts*

DstT0*

SrcT0*'
_output_shapes
:         q
LogisticRegression_1/Cast_43Cast
active_max*

DstT0*

SrcT0*'
_output_shapes
:         q
LogisticRegression_1/Cast_44Cast
active_min*

DstT0*

SrcT0*'
_output_shapes
:         r
LogisticRegression_1/Cast_45Castactive_mean*

DstT0*

SrcT0*'
_output_shapes
:         q
LogisticRegression_1/Cast_46Cast
active_std*

DstT0*

SrcT0*'
_output_shapes
:         o
LogisticRegression_1/Cast_47Castidle_max*

DstT0*

SrcT0*'
_output_shapes
:         o
LogisticRegression_1/Cast_48Castidle_min*

DstT0*

SrcT0*'
_output_shapes
:         p
LogisticRegression_1/Cast_49Cast	idle_mean*

DstT0*

SrcT0*'
_output_shapes
:         o
LogisticRegression_1/Cast_50Castidle_std*

DstT0*

SrcT0*'
_output_shapes
:         u
LogisticRegression_1/Cast_51Castfwd_byts_b_avg*

DstT0*

SrcT0*'
_output_shapes
:         u
LogisticRegression_1/Cast_52Castfwd_pkts_b_avg*

DstT0*

SrcT0*'
_output_shapes
:         u
LogisticRegression_1/Cast_53Castbwd_byts_b_avg*

DstT0*

SrcT0*'
_output_shapes
:         u
LogisticRegression_1/Cast_54Castbwd_pkts_b_avg*

DstT0*

SrcT0*'
_output_shapes
:         w
LogisticRegression_1/Cast_55Castfwd_blk_rate_avg*

DstT0*

SrcT0*'
_output_shapes
:         w
LogisticRegression_1/Cast_56Castbwd_blk_rate_avg*

DstT0*

SrcT0*'
_output_shapes
:         z
'LogisticRegression_1/concatenate_1/CastCastsrc_port*

DstT0*

SrcT0*'
_output_shapes
:         |
)LogisticRegression_1/concatenate_1/Cast_1Castdst_port*

DstT0*

SrcT0*'
_output_shapes
:         |
)LogisticRegression_1/concatenate_1/Cast_2Castprotocol*

DstT0*

SrcT0*'
_output_shapes
:         ~
)LogisticRegression_1/concatenate_1/Cast_3Cast
pkt_length*

DstT0*

SrcT0*'
_output_shapes
:         В
)LogisticRegression_1/concatenate_1/Cast_4Castfwd_header_len*

DstT0*

SrcT0*'
_output_shapes
:         В
)LogisticRegression_1/concatenate_1/Cast_5Castbwd_header_len*

DstT0*

SrcT0*'
_output_shapes
:         Б
)LogisticRegression_1/concatenate_1/Cast_6Castfwd_psh_flags*

DstT0*

SrcT0*'
_output_shapes
:         Б
)LogisticRegression_1/concatenate_1/Cast_7Castbwd_psh_flags*

DstT0*

SrcT0*'
_output_shapes
:         Б
)LogisticRegression_1/concatenate_1/Cast_8Castfwd_urg_flags*

DstT0*

SrcT0*'
_output_shapes
:         Б
)LogisticRegression_1/concatenate_1/Cast_9Castbwd_urg_flags*

DstT0*

SrcT0*'
_output_shapes
:         Б
*LogisticRegression_1/concatenate_1/Cast_10Castfin_flag_cnt*

DstT0*

SrcT0*'
_output_shapes
:         Б
*LogisticRegression_1/concatenate_1/Cast_11Castsyn_flag_cnt*

DstT0*

SrcT0*'
_output_shapes
:         Б
*LogisticRegression_1/concatenate_1/Cast_12Castrst_flag_cnt*

DstT0*

SrcT0*'
_output_shapes
:         Б
*LogisticRegression_1/concatenate_1/Cast_13Castpsh_flag_cnt*

DstT0*

SrcT0*'
_output_shapes
:         Б
*LogisticRegression_1/concatenate_1/Cast_14Castack_flag_cnt*

DstT0*

SrcT0*'
_output_shapes
:         Б
*LogisticRegression_1/concatenate_1/Cast_15Casturg_flag_cnt*

DstT0*

SrcT0*'
_output_shapes
:         Б
*LogisticRegression_1/concatenate_1/Cast_16Castece_flag_cnt*

DstT0*

SrcT0*'
_output_shapes
:         y
.LogisticRegression_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         ╙
)LogisticRegression_1/concatenate_1/concatConcatV2+LogisticRegression_1/concatenate_1/Cast:y:0-LogisticRegression_1/concatenate_1/Cast_1:y:0-LogisticRegression_1/concatenate_1/Cast_2:y:0-LogisticRegression_1/concatenate_1/Cast_3:y:0LogisticRegression_1/Cast:y:0LogisticRegression_1/Cast_1:y:0LogisticRegression_1/Cast_2:y:0LogisticRegression_1/Cast_3:y:0LogisticRegression_1/Cast_4:y:0LogisticRegression_1/Cast_5:y:0LogisticRegression_1/Cast_6:y:0LogisticRegression_1/Cast_7:y:0LogisticRegression_1/Cast_8:y:0LogisticRegression_1/Cast_9:y:0 LogisticRegression_1/Cast_10:y:0 LogisticRegression_1/Cast_11:y:0 LogisticRegression_1/Cast_12:y:0 LogisticRegression_1/Cast_13:y:0 LogisticRegression_1/Cast_14:y:0 LogisticRegression_1/Cast_15:y:0 LogisticRegression_1/Cast_16:y:0 LogisticRegression_1/Cast_17:y:0 LogisticRegression_1/Cast_18:y:0 LogisticRegression_1/Cast_19:y:0 LogisticRegression_1/Cast_20:y:0 LogisticRegression_1/Cast_21:y:0 LogisticRegression_1/Cast_22:y:0-LogisticRegression_1/concatenate_1/Cast_4:y:0-LogisticRegression_1/concatenate_1/Cast_5:y:0 LogisticRegression_1/Cast_23:y:0 LogisticRegression_1/Cast_24:y:0 LogisticRegression_1/Cast_25:y:0 LogisticRegression_1/Cast_26:y:0 LogisticRegression_1/Cast_27:y:0 LogisticRegression_1/Cast_28:y:0 LogisticRegression_1/Cast_29:y:0 LogisticRegression_1/Cast_30:y:0 LogisticRegression_1/Cast_31:y:0 LogisticRegression_1/Cast_32:y:0 LogisticRegression_1/Cast_33:y:0 LogisticRegression_1/Cast_34:y:0 LogisticRegression_1/Cast_35:y:0 LogisticRegression_1/Cast_36:y:0 LogisticRegression_1/Cast_37:y:0 LogisticRegression_1/Cast_38:y:0-LogisticRegression_1/concatenate_1/Cast_6:y:0-LogisticRegression_1/concatenate_1/Cast_7:y:0-LogisticRegression_1/concatenate_1/Cast_8:y:0-LogisticRegression_1/concatenate_1/Cast_9:y:0.LogisticRegression_1/concatenate_1/Cast_10:y:0.LogisticRegression_1/concatenate_1/Cast_11:y:0.LogisticRegression_1/concatenate_1/Cast_12:y:0.LogisticRegression_1/concatenate_1/Cast_13:y:0.LogisticRegression_1/concatenate_1/Cast_14:y:0.LogisticRegression_1/concatenate_1/Cast_15:y:0.LogisticRegression_1/concatenate_1/Cast_16:y:0 LogisticRegression_1/Cast_39:y:0 LogisticRegression_1/Cast_40:y:0 LogisticRegression_1/Cast_41:y:0 LogisticRegression_1/Cast_42:y:0 LogisticRegression_1/Cast_43:y:0 LogisticRegression_1/Cast_44:y:0 LogisticRegression_1/Cast_45:y:0 LogisticRegression_1/Cast_46:y:0 LogisticRegression_1/Cast_47:y:0 LogisticRegression_1/Cast_48:y:0 LogisticRegression_1/Cast_49:y:0 LogisticRegression_1/Cast_50:y:0 LogisticRegression_1/Cast_51:y:0 LogisticRegression_1/Cast_52:y:0 LogisticRegression_1/Cast_53:y:0 LogisticRegression_1/Cast_54:y:0 LogisticRegression_1/Cast_55:y:0 LogisticRegression_1/Cast_56:y:07LogisticRegression_1/concatenate_1/concat/axis:output:0*
NJ*
T0*'
_output_shapes
:         J╛
:LogisticRegression_1/Network_1/dense_1/Cast/ReadVariableOpReadVariableOpClogisticregression_1_network_1_dense_1_cast_readvariableop_resource*
_output_shapes

:J*
dtype0с
-LogisticRegression_1/Network_1/dense_1/MatMulMatMul2LogisticRegression_1/concatenate_1/concat:output:0BLogisticRegression_1/Network_1/dense_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ╕
9LogisticRegression_1/Network_1/dense_1/Add/ReadVariableOpReadVariableOpBlogisticregression_1_network_1_dense_1_add_readvariableop_resource*
_output_shapes
:*
dtype0с
*LogisticRegression_1/Network_1/dense_1/AddAddV27LogisticRegression_1/Network_1/dense_1/MatMul:product:0ALogisticRegression_1/Network_1/dense_1/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ы
.LogisticRegression_1/Network_1/dense_1/SigmoidSigmoid.LogisticRegression_1/Network_1/dense_1/Add:z:0*
T0*'
_output_shapes
:         Б
IdentityIdentity2LogisticRegression_1/Network_1/dense_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         Ы
NoOpNoOp:^LogisticRegression_1/Network_1/dense_1/Add/ReadVariableOp;^LogisticRegression_1/Network_1/dense_1/Cast/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ч
_input_shapesЕ
В:         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         : : 2v
9LogisticRegression_1/Network_1/dense_1/Add/ReadVariableOp9LogisticRegression_1/Network_1/dense_1/Add/ReadVariableOp2x
:LogisticRegression_1/Network_1/dense_1/Cast/ReadVariableOp:LogisticRegression_1/Network_1/dense_1/Cast/ReadVariableOp:(K$
"
_user_specified_name
resource:(J$
"
_user_specified_name
resource:UIQ
'
_output_shapes
:         
&
_user_specified_nameurg_flag_cnt:XHT
'
_output_shapes
:         
)
_user_specified_nametotlen_fwd_pkts:XGT
'
_output_shapes
:         
)
_user_specified_nametotlen_bwd_pkts:UFQ
'
_output_shapes
:         
&
_user_specified_nametot_fwd_pkts:UEQ
'
_output_shapes
:         
&
_user_specified_nametot_bwd_pkts:UDQ
'
_output_shapes
:         
&
_user_specified_namesyn_flag_cnt:QCM
'
_output_shapes
:         
"
_user_specified_name
src_port:UBQ
'
_output_shapes
:         
&
_user_specified_namerst_flag_cnt:UAQ
'
_output_shapes
:         
&
_user_specified_namepsh_flag_cnt:Q@M
'
_output_shapes
:         
"
_user_specified_name
protocol:U?Q
'
_output_shapes
:         
&
_user_specified_namepkt_size_avg:S>O
'
_output_shapes
:         
$
_user_specified_name
pkt_length:T=P
'
_output_shapes
:         
%
_user_specified_namepkt_len_var:T<P
'
_output_shapes
:         
%
_user_specified_namepkt_len_std:T;P
'
_output_shapes
:         
%
_user_specified_namepkt_len_min:U:Q
'
_output_shapes
:         
&
_user_specified_namepkt_len_mean:T9P
'
_output_shapes
:         
%
_user_specified_namepkt_len_max:Z8V
'
_output_shapes
:         
+
_user_specified_nameinit_fwd_win_byts:Z7V
'
_output_shapes
:         
+
_user_specified_nameinit_bwd_win_byts:M6I
'
_output_shapes
:         

_user_specified_nameinfo:Q5M
'
_output_shapes
:         
"
_user_specified_name
idle_std:Q4M
'
_output_shapes
:         
"
_user_specified_name
idle_min:R3N
'
_output_shapes
:         
#
_user_specified_name	idle_mean:Q2M
'
_output_shapes
:         
"
_user_specified_name
idle_max:V1R
'
_output_shapes
:         
'
_user_specified_namefwd_urg_flags:Y0U
'
_output_shapes
:         
*
_user_specified_namefwd_seg_size_min:V/R
'
_output_shapes
:         
'
_user_specified_namefwd_psh_flags:S.O
'
_output_shapes
:         
$
_user_specified_name
fwd_pkts_s:W-S
'
_output_shapes
:         
(
_user_specified_namefwd_pkts_b_avg:X,T
'
_output_shapes
:         
)
_user_specified_namefwd_pkt_len_std:X+T
'
_output_shapes
:         
)
_user_specified_namefwd_pkt_len_min:Y*U
'
_output_shapes
:         
*
_user_specified_namefwd_pkt_len_mean:X)T
'
_output_shapes
:         
)
_user_specified_namefwd_pkt_len_max:T(P
'
_output_shapes
:         
%
_user_specified_namefwd_iat_tot:T'P
'
_output_shapes
:         
%
_user_specified_namefwd_iat_std:T&P
'
_output_shapes
:         
%
_user_specified_namefwd_iat_min:U%Q
'
_output_shapes
:         
&
_user_specified_namefwd_iat_mean:T$P
'
_output_shapes
:         
%
_user_specified_namefwd_iat_max:W#S
'
_output_shapes
:         
(
_user_specified_namefwd_header_len:W"S
'
_output_shapes
:         
(
_user_specified_namefwd_byts_b_avg:Y!U
'
_output_shapes
:         
*
_user_specified_namefwd_blk_rate_avg:Z V
'
_output_shapes
:         
+
_user_specified_namefwd_act_data_pkts:TP
'
_output_shapes
:         
%
_user_specified_nameflow_pkts_s:UQ
'
_output_shapes
:         
&
_user_specified_nameflow_iat_std:UQ
'
_output_shapes
:         
&
_user_specified_nameflow_iat_min:VR
'
_output_shapes
:         
'
_user_specified_nameflow_iat_mean:UQ
'
_output_shapes
:         
&
_user_specified_nameflow_iat_max:VR
'
_output_shapes
:         
'
_user_specified_nameflow_duration:TP
'
_output_shapes
:         
%
_user_specified_nameflow_byts_s:UQ
'
_output_shapes
:         
&
_user_specified_namefin_flag_cnt:UQ
'
_output_shapes
:         
&
_user_specified_nameece_flag_cnt:QM
'
_output_shapes
:         
"
_user_specified_name
dst_port:VR
'
_output_shapes
:         
'
_user_specified_namedown_up_ratio:VR
'
_output_shapes
:         
'
_user_specified_namebwd_urg_flags:VR
'
_output_shapes
:         
'
_user_specified_namebwd_psh_flags:SO
'
_output_shapes
:         
$
_user_specified_name
bwd_pkts_s:WS
'
_output_shapes
:         
(
_user_specified_namebwd_pkts_b_avg:XT
'
_output_shapes
:         
)
_user_specified_namebwd_pkt_len_std:XT
'
_output_shapes
:         
)
_user_specified_namebwd_pkt_len_min:YU
'
_output_shapes
:         
*
_user_specified_namebwd_pkt_len_mean:XT
'
_output_shapes
:         
)
_user_specified_namebwd_pkt_len_max:TP
'
_output_shapes
:         
%
_user_specified_namebwd_iat_tot:TP
'
_output_shapes
:         
%
_user_specified_namebwd_iat_std:T
P
'
_output_shapes
:         
%
_user_specified_namebwd_iat_min:U	Q
'
_output_shapes
:         
&
_user_specified_namebwd_iat_mean:TP
'
_output_shapes
:         
%
_user_specified_namebwd_iat_max:WS
'
_output_shapes
:         
(
_user_specified_namebwd_header_len:WS
'
_output_shapes
:         
(
_user_specified_namebwd_byts_b_avg:YU
'
_output_shapes
:         
*
_user_specified_namebwd_blk_rate_avg:SO
'
_output_shapes
:         
$
_user_specified_name
active_std:SO
'
_output_shapes
:         
$
_user_specified_name
active_min:TP
'
_output_shapes
:         
%
_user_specified_nameactive_mean:SO
'
_output_shapes
:         
$
_user_specified_name
active_max:U Q
'
_output_shapes
:         
&
_user_specified_nameack_flag_cnt
░
ч
!__inference__traced_restore_18074
file_prefix-
assignvariableop_variable_1:J)
assignvariableop_1_variable:;
)assignvariableop_2_network_dense_kernel_1:J5
'assignvariableop_3_network_dense_bias_1:

identity_5ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_2вAssignVariableOp_3╖
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*▌
value╙B╨B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHz
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B ╖
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOpAssignVariableOpassignvariableop_variable_1Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_1AssignVariableOpassignvariableop_1_variableIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_2AssignVariableOp)assignvariableop_2_network_dense_kernel_1Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_3AssignVariableOp'assignvariableop_3_network_dense_bias_1Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 м

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_5IdentityIdentity_4:output:0^NoOp_1*
T0*
_output_shapes
: v
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*
_output_shapes
 "!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32$
AssignVariableOpAssignVariableOp:40
.
_user_specified_nameNetwork/dense/bias_1:62
0
_user_specified_nameNetwork/dense/kernel_1:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"┌L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ж$
serveЬ$
;
ack_flag_cnt+
serve_ack_flag_cnt:0         
7

active_max)
serve_active_max:0         
9
active_mean*
serve_active_mean:0         
7

active_min)
serve_active_min:0         
7

active_std)
serve_active_std:0         
C
bwd_blk_rate_avg/
serve_bwd_blk_rate_avg:0         
?
bwd_byts_b_avg-
serve_bwd_byts_b_avg:0         
?
bwd_header_len-
serve_bwd_header_len:0         
9
bwd_iat_max*
serve_bwd_iat_max:0         
;
bwd_iat_mean+
serve_bwd_iat_mean:0         
9
bwd_iat_min*
serve_bwd_iat_min:0         
9
bwd_iat_std*
serve_bwd_iat_std:0         
9
bwd_iat_tot*
serve_bwd_iat_tot:0         
A
bwd_pkt_len_max.
serve_bwd_pkt_len_max:0         
C
bwd_pkt_len_mean/
serve_bwd_pkt_len_mean:0         
A
bwd_pkt_len_min.
serve_bwd_pkt_len_min:0         
A
bwd_pkt_len_std.
serve_bwd_pkt_len_std:0         
?
bwd_pkts_b_avg-
serve_bwd_pkts_b_avg:0         
7

bwd_pkts_s)
serve_bwd_pkts_s:0         
=
bwd_psh_flags,
serve_bwd_psh_flags:0         
=
bwd_urg_flags,
serve_bwd_urg_flags:0         
=
down_up_ratio,
serve_down_up_ratio:0         
3
dst_port'
serve_dst_port:0         
;
ece_flag_cnt+
serve_ece_flag_cnt:0         
;
fin_flag_cnt+
serve_fin_flag_cnt:0         
9
flow_byts_s*
serve_flow_byts_s:0         
=
flow_duration,
serve_flow_duration:0         
;
flow_iat_max+
serve_flow_iat_max:0         
=
flow_iat_mean,
serve_flow_iat_mean:0         
;
flow_iat_min+
serve_flow_iat_min:0         
;
flow_iat_std+
serve_flow_iat_std:0         
9
flow_pkts_s*
serve_flow_pkts_s:0         
E
fwd_act_data_pkts0
serve_fwd_act_data_pkts:0         
C
fwd_blk_rate_avg/
serve_fwd_blk_rate_avg:0         
?
fwd_byts_b_avg-
serve_fwd_byts_b_avg:0         
?
fwd_header_len-
serve_fwd_header_len:0         
9
fwd_iat_max*
serve_fwd_iat_max:0         
;
fwd_iat_mean+
serve_fwd_iat_mean:0         
9
fwd_iat_min*
serve_fwd_iat_min:0         
9
fwd_iat_std*
serve_fwd_iat_std:0         
9
fwd_iat_tot*
serve_fwd_iat_tot:0         
A
fwd_pkt_len_max.
serve_fwd_pkt_len_max:0         
C
fwd_pkt_len_mean/
serve_fwd_pkt_len_mean:0         
A
fwd_pkt_len_min.
serve_fwd_pkt_len_min:0         
A
fwd_pkt_len_std.
serve_fwd_pkt_len_std:0         
?
fwd_pkts_b_avg-
serve_fwd_pkts_b_avg:0         
7

fwd_pkts_s)
serve_fwd_pkts_s:0         
=
fwd_psh_flags,
serve_fwd_psh_flags:0         
C
fwd_seg_size_min/
serve_fwd_seg_size_min:0         
=
fwd_urg_flags,
serve_fwd_urg_flags:0         
3
idle_max'
serve_idle_max:0         
5
	idle_mean(
serve_idle_mean:0         
3
idle_min'
serve_idle_min:0         
3
idle_std'
serve_idle_std:0         
+
info#
serve_info:0         
E
init_bwd_win_byts0
serve_init_bwd_win_byts:0         
E
init_fwd_win_byts0
serve_init_fwd_win_byts:0         
9
pkt_len_max*
serve_pkt_len_max:0         
;
pkt_len_mean+
serve_pkt_len_mean:0         
9
pkt_len_min*
serve_pkt_len_min:0         
9
pkt_len_std*
serve_pkt_len_std:0         
9
pkt_len_var*
serve_pkt_len_var:0         
7

pkt_length)
serve_pkt_length:0         
;
pkt_size_avg+
serve_pkt_size_avg:0         
3
protocol'
serve_protocol:0         
;
psh_flag_cnt+
serve_psh_flag_cnt:0         
;
rst_flag_cnt+
serve_rst_flag_cnt:0         
3
src_port'
serve_src_port:0         
;
syn_flag_cnt+
serve_syn_flag_cnt:0         
;
tot_bwd_pkts+
serve_tot_bwd_pkts:0         
;
tot_fwd_pkts+
serve_tot_fwd_pkts:0         
A
totlen_bwd_pkts.
serve_totlen_bwd_pkts:0         
A
totlen_fwd_pkts.
serve_totlen_fwd_pkts:0         
;
urg_flag_cnt+
serve_urg_flag_cnt:0         <
output_00
StatefulPartitionedCall:0         tensorflow/serving/predict*Ц*
serving_defaultВ*
E
ack_flag_cnt5
serving_default_ack_flag_cnt:0         
A

active_max3
serving_default_active_max:0         
C
active_mean4
serving_default_active_mean:0         
A

active_min3
serving_default_active_min:0         
A

active_std3
serving_default_active_std:0         
M
bwd_blk_rate_avg9
"serving_default_bwd_blk_rate_avg:0         
I
bwd_byts_b_avg7
 serving_default_bwd_byts_b_avg:0         
I
bwd_header_len7
 serving_default_bwd_header_len:0         
C
bwd_iat_max4
serving_default_bwd_iat_max:0         
E
bwd_iat_mean5
serving_default_bwd_iat_mean:0         
C
bwd_iat_min4
serving_default_bwd_iat_min:0         
C
bwd_iat_std4
serving_default_bwd_iat_std:0         
C
bwd_iat_tot4
serving_default_bwd_iat_tot:0         
K
bwd_pkt_len_max8
!serving_default_bwd_pkt_len_max:0         
M
bwd_pkt_len_mean9
"serving_default_bwd_pkt_len_mean:0         
K
bwd_pkt_len_min8
!serving_default_bwd_pkt_len_min:0         
K
bwd_pkt_len_std8
!serving_default_bwd_pkt_len_std:0         
I
bwd_pkts_b_avg7
 serving_default_bwd_pkts_b_avg:0         
A

bwd_pkts_s3
serving_default_bwd_pkts_s:0         
G
bwd_psh_flags6
serving_default_bwd_psh_flags:0         
G
bwd_urg_flags6
serving_default_bwd_urg_flags:0         
G
down_up_ratio6
serving_default_down_up_ratio:0         
=
dst_port1
serving_default_dst_port:0         
E
ece_flag_cnt5
serving_default_ece_flag_cnt:0         
E
fin_flag_cnt5
serving_default_fin_flag_cnt:0         
C
flow_byts_s4
serving_default_flow_byts_s:0         
G
flow_duration6
serving_default_flow_duration:0         
E
flow_iat_max5
serving_default_flow_iat_max:0         
G
flow_iat_mean6
serving_default_flow_iat_mean:0         
E
flow_iat_min5
serving_default_flow_iat_min:0         
E
flow_iat_std5
serving_default_flow_iat_std:0         
C
flow_pkts_s4
serving_default_flow_pkts_s:0         
O
fwd_act_data_pkts:
#serving_default_fwd_act_data_pkts:0         
M
fwd_blk_rate_avg9
"serving_default_fwd_blk_rate_avg:0         
I
fwd_byts_b_avg7
 serving_default_fwd_byts_b_avg:0         
I
fwd_header_len7
 serving_default_fwd_header_len:0         
C
fwd_iat_max4
serving_default_fwd_iat_max:0         
E
fwd_iat_mean5
serving_default_fwd_iat_mean:0         
C
fwd_iat_min4
serving_default_fwd_iat_min:0         
C
fwd_iat_std4
serving_default_fwd_iat_std:0         
C
fwd_iat_tot4
serving_default_fwd_iat_tot:0         
K
fwd_pkt_len_max8
!serving_default_fwd_pkt_len_max:0         
M
fwd_pkt_len_mean9
"serving_default_fwd_pkt_len_mean:0         
K
fwd_pkt_len_min8
!serving_default_fwd_pkt_len_min:0         
K
fwd_pkt_len_std8
!serving_default_fwd_pkt_len_std:0         
I
fwd_pkts_b_avg7
 serving_default_fwd_pkts_b_avg:0         
A

fwd_pkts_s3
serving_default_fwd_pkts_s:0         
G
fwd_psh_flags6
serving_default_fwd_psh_flags:0         
M
fwd_seg_size_min9
"serving_default_fwd_seg_size_min:0         
G
fwd_urg_flags6
serving_default_fwd_urg_flags:0         
=
idle_max1
serving_default_idle_max:0         
?
	idle_mean2
serving_default_idle_mean:0         
=
idle_min1
serving_default_idle_min:0         
=
idle_std1
serving_default_idle_std:0         
5
info-
serving_default_info:0         
O
init_bwd_win_byts:
#serving_default_init_bwd_win_byts:0         
O
init_fwd_win_byts:
#serving_default_init_fwd_win_byts:0         
C
pkt_len_max4
serving_default_pkt_len_max:0         
E
pkt_len_mean5
serving_default_pkt_len_mean:0         
C
pkt_len_min4
serving_default_pkt_len_min:0         
C
pkt_len_std4
serving_default_pkt_len_std:0         
C
pkt_len_var4
serving_default_pkt_len_var:0         
A

pkt_length3
serving_default_pkt_length:0         
E
pkt_size_avg5
serving_default_pkt_size_avg:0         
=
protocol1
serving_default_protocol:0         
E
psh_flag_cnt5
serving_default_psh_flag_cnt:0         
E
rst_flag_cnt5
serving_default_rst_flag_cnt:0         
=
src_port1
serving_default_src_port:0         
E
syn_flag_cnt5
serving_default_syn_flag_cnt:0         
E
tot_bwd_pkts5
serving_default_tot_bwd_pkts:0         
E
tot_fwd_pkts5
serving_default_tot_fwd_pkts:0         
K
totlen_bwd_pkts8
!serving_default_totlen_bwd_pkts:0         
K
totlen_fwd_pkts8
!serving_default_totlen_fwd_pkts:0         
E
urg_flag_cnt5
serving_default_urg_flag_cnt:0         >
output_02
StatefulPartitionedCall_1:0         tensorflow/serving/predict:яи
д
	variables
trainable_variables
non_trainable_variables
_all_variables
_misc_assets
	serve

signatures"
_generic_user_object
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
■
trace_02с
__inference___call___17686┬
С▓Н
FullArgSpec
argsЪ

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *лвз
дЪа
&К#
ack_flag_cnt         
$К!

active_max         
%К"
active_mean         
$К!

active_min         
$К!

active_std         
*К'
bwd_blk_rate_avg         
(К%
bwd_byts_b_avg         
(К%
bwd_header_len         
%К"
bwd_iat_max         
&К#
bwd_iat_mean         
%К"
bwd_iat_min         
%К"
bwd_iat_std         
%К"
bwd_iat_tot         
)К&
bwd_pkt_len_max         
*К'
bwd_pkt_len_mean         
)К&
bwd_pkt_len_min         
)К&
bwd_pkt_len_std         
(К%
bwd_pkts_b_avg         
$К!

bwd_pkts_s         
'К$
bwd_psh_flags         
'К$
bwd_urg_flags         
'К$
down_up_ratio         
"К
dst_port         
&К#
ece_flag_cnt         
&К#
fin_flag_cnt         
%К"
flow_byts_s         
'К$
flow_duration         
&К#
flow_iat_max         
'К$
flow_iat_mean         
&К#
flow_iat_min         
&К#
flow_iat_std         
%К"
flow_pkts_s         
+К(
fwd_act_data_pkts         
*К'
fwd_blk_rate_avg         
(К%
fwd_byts_b_avg         
(К%
fwd_header_len         
%К"
fwd_iat_max         
&К#
fwd_iat_mean         
%К"
fwd_iat_min         
%К"
fwd_iat_std         
%К"
fwd_iat_tot         
)К&
fwd_pkt_len_max         
*К'
fwd_pkt_len_mean         
)К&
fwd_pkt_len_min         
)К&
fwd_pkt_len_std         
(К%
fwd_pkts_b_avg         
$К!

fwd_pkts_s         
'К$
fwd_psh_flags         
*К'
fwd_seg_size_min         
'К$
fwd_urg_flags         
"К
idle_max         
#К 
	idle_mean         
"К
idle_min         
"К
idle_std         
К
info         
+К(
init_bwd_win_byts         
+К(
init_fwd_win_byts         
%К"
pkt_len_max         
&К#
pkt_len_mean         
%К"
pkt_len_min         
%К"
pkt_len_std         
%К"
pkt_len_var         
$К!

pkt_length         
&К#
pkt_size_avg         
"К
protocol         
&К#
psh_flag_cnt         
&К#
rst_flag_cnt         
"К
src_port         
&К#
syn_flag_cnt         
&К#
tot_bwd_pkts         
&К#
tot_fwd_pkts         
)К&
totlen_bwd_pkts         
)К&
totlen_fwd_pkts         
&К#
urg_flag_cnt         ztrace_0
7
	serve
serving_default"
signature_map
&:$J2Network/dense/kernel
 :2Network/dense/bias
&:$J2Network/dense/kernel
 :2Network/dense/bias
╪	B╒	
__inference___call___17686ack_flag_cnt
active_maxactive_mean
active_min
active_stdbwd_blk_rate_avgbwd_byts_b_avgbwd_header_lenbwd_iat_maxbwd_iat_meanbwd_iat_minbwd_iat_stdbwd_iat_totbwd_pkt_len_maxbwd_pkt_len_meanbwd_pkt_len_minbwd_pkt_len_stdbwd_pkts_b_avg
bwd_pkts_sbwd_psh_flagsbwd_urg_flagsdown_up_ratiodst_portece_flag_cntfin_flag_cntflow_byts_sflow_durationflow_iat_maxflow_iat_meanflow_iat_minflow_iat_stdflow_pkts_sfwd_act_data_pktsfwd_blk_rate_avgfwd_byts_b_avgfwd_header_lenfwd_iat_maxfwd_iat_meanfwd_iat_minfwd_iat_stdfwd_iat_totfwd_pkt_len_maxfwd_pkt_len_meanfwd_pkt_len_minfwd_pkt_len_stdfwd_pkts_b_avg
fwd_pkts_sfwd_psh_flagsfwd_seg_size_minfwd_urg_flagsidle_max	idle_meanidle_minidle_stdinfoinit_bwd_win_bytsinit_fwd_win_bytspkt_len_maxpkt_len_meanpkt_len_minpkt_len_stdpkt_len_var
pkt_lengthpkt_size_avgprotocolpsh_flag_cntrst_flag_cntsrc_portsyn_flag_cnttot_bwd_pktstot_fwd_pktstotlen_bwd_pktstotlen_fwd_pktsurg_flag_cntJ"Ш
С▓Н
FullArgSpec
argsЪ

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
СBО
,__inference_signature_wrapper___call___17769ack_flag_cnt
active_maxactive_mean
active_min
active_stdbwd_blk_rate_avgbwd_byts_b_avgbwd_header_lenbwd_iat_maxbwd_iat_meanbwd_iat_minbwd_iat_stdbwd_iat_totbwd_pkt_len_maxbwd_pkt_len_meanbwd_pkt_len_minbwd_pkt_len_stdbwd_pkts_b_avg
bwd_pkts_sbwd_psh_flagsbwd_urg_flagsdown_up_ratiodst_portece_flag_cntfin_flag_cntflow_byts_sflow_durationflow_iat_maxflow_iat_meanflow_iat_minflow_iat_stdflow_pkts_sfwd_act_data_pktsfwd_blk_rate_avgfwd_byts_b_avgfwd_header_lenfwd_iat_maxfwd_iat_meanfwd_iat_minfwd_iat_stdfwd_iat_totfwd_pkt_len_maxfwd_pkt_len_meanfwd_pkt_len_minfwd_pkt_len_stdfwd_pkts_b_avg
fwd_pkts_sfwd_psh_flagsfwd_seg_size_minfwd_urg_flagsidle_max	idle_meanidle_minidle_stdinfoinit_bwd_win_bytsinit_fwd_win_bytspkt_len_maxpkt_len_meanpkt_len_minpkt_len_stdpkt_len_var
pkt_lengthpkt_size_avgprotocolpsh_flag_cntrst_flag_cntsrc_portsyn_flag_cnttot_bwd_pktstot_fwd_pktstotlen_bwd_pktstotlen_fwd_pktsurg_flag_cnt"┴

║
▓╢

FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 ├	

kwonlyargs┤	Ъ░	
jack_flag_cnt
j
active_max
jactive_mean
j
active_min
j
active_std
jbwd_blk_rate_avg
jbwd_byts_b_avg
jbwd_header_len
jbwd_iat_max
jbwd_iat_mean
jbwd_iat_min
jbwd_iat_std
jbwd_iat_tot
jbwd_pkt_len_max
jbwd_pkt_len_mean
jbwd_pkt_len_min
jbwd_pkt_len_std
jbwd_pkts_b_avg
j
bwd_pkts_s
jbwd_psh_flags
jbwd_urg_flags
jdown_up_ratio

jdst_port
jece_flag_cnt
jfin_flag_cnt
jflow_byts_s
jflow_duration
jflow_iat_max
jflow_iat_mean
jflow_iat_min
jflow_iat_std
jflow_pkts_s
jfwd_act_data_pkts
jfwd_blk_rate_avg
jfwd_byts_b_avg
jfwd_header_len
jfwd_iat_max
jfwd_iat_mean
jfwd_iat_min
jfwd_iat_std
jfwd_iat_tot
jfwd_pkt_len_max
jfwd_pkt_len_mean
jfwd_pkt_len_min
jfwd_pkt_len_std
jfwd_pkts_b_avg
j
fwd_pkts_s
jfwd_psh_flags
jfwd_seg_size_min
jfwd_urg_flags

jidle_max
j	idle_mean

jidle_min

jidle_std
jinfo
jinit_bwd_win_byts
jinit_fwd_win_byts
jpkt_len_max
jpkt_len_mean
jpkt_len_min
jpkt_len_std
jpkt_len_var
j
pkt_length
jpkt_size_avg

jprotocol
jpsh_flag_cnt
jrst_flag_cnt

jsrc_port
jsyn_flag_cnt
jtot_bwd_pkts
jtot_fwd_pkts
jtotlen_bwd_pkts
jtotlen_fwd_pkts
jurg_flag_cnt
kwonlydefaults
 
annotationsк *
 
СBО
,__inference_signature_wrapper___call___17851ack_flag_cnt
active_maxactive_mean
active_min
active_stdbwd_blk_rate_avgbwd_byts_b_avgbwd_header_lenbwd_iat_maxbwd_iat_meanbwd_iat_minbwd_iat_stdbwd_iat_totbwd_pkt_len_maxbwd_pkt_len_meanbwd_pkt_len_minbwd_pkt_len_stdbwd_pkts_b_avg
bwd_pkts_sbwd_psh_flagsbwd_urg_flagsdown_up_ratiodst_portece_flag_cntfin_flag_cntflow_byts_sflow_durationflow_iat_maxflow_iat_meanflow_iat_minflow_iat_stdflow_pkts_sfwd_act_data_pktsfwd_blk_rate_avgfwd_byts_b_avgfwd_header_lenfwd_iat_maxfwd_iat_meanfwd_iat_minfwd_iat_stdfwd_iat_totfwd_pkt_len_maxfwd_pkt_len_meanfwd_pkt_len_minfwd_pkt_len_stdfwd_pkts_b_avg
fwd_pkts_sfwd_psh_flagsfwd_seg_size_minfwd_urg_flagsidle_max	idle_meanidle_minidle_stdinfoinit_bwd_win_bytsinit_fwd_win_bytspkt_len_maxpkt_len_meanpkt_len_minpkt_len_stdpkt_len_var
pkt_lengthpkt_size_avgprotocolpsh_flag_cntrst_flag_cntsrc_portsyn_flag_cnttot_bwd_pktstot_fwd_pktstotlen_bwd_pktstotlen_fwd_pktsurg_flag_cnt"┴

║
▓╢

FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 ├	

kwonlyargs┤	Ъ░	
jack_flag_cnt
j
active_max
jactive_mean
j
active_min
j
active_std
jbwd_blk_rate_avg
jbwd_byts_b_avg
jbwd_header_len
jbwd_iat_max
jbwd_iat_mean
jbwd_iat_min
jbwd_iat_std
jbwd_iat_tot
jbwd_pkt_len_max
jbwd_pkt_len_mean
jbwd_pkt_len_min
jbwd_pkt_len_std
jbwd_pkts_b_avg
j
bwd_pkts_s
jbwd_psh_flags
jbwd_urg_flags
jdown_up_ratio

jdst_port
jece_flag_cnt
jfin_flag_cnt
jflow_byts_s
jflow_duration
jflow_iat_max
jflow_iat_mean
jflow_iat_min
jflow_iat_std
jflow_pkts_s
jfwd_act_data_pkts
jfwd_blk_rate_avg
jfwd_byts_b_avg
jfwd_header_len
jfwd_iat_max
jfwd_iat_mean
jfwd_iat_min
jfwd_iat_std
jfwd_iat_tot
jfwd_pkt_len_max
jfwd_pkt_len_mean
jfwd_pkt_len_min
jfwd_pkt_len_std
jfwd_pkts_b_avg
j
fwd_pkts_s
jfwd_psh_flags
jfwd_seg_size_min
jfwd_urg_flags

jidle_max
j	idle_mean

jidle_min

jidle_std
jinfo
jinit_bwd_win_byts
jinit_fwd_win_byts
jpkt_len_max
jpkt_len_mean
jpkt_len_min
jpkt_len_std
jpkt_len_var
j
pkt_length
jpkt_size_avg

jprotocol
jpsh_flag_cnt
jrst_flag_cnt

jsrc_port
jsyn_flag_cnt
jtot_bwd_pkts
jtot_fwd_pkts
jtotlen_bwd_pkts
jtotlen_fwd_pkts
jurg_flag_cnt
kwonlydefaults
 
annotationsк *
 А
__inference___call___17686с	╖в│
лвз
дЪа
&К#
ack_flag_cnt         
$К!

active_max         
%К"
active_mean         
$К!

active_min         
$К!

active_std         
*К'
bwd_blk_rate_avg         
(К%
bwd_byts_b_avg         
(К%
bwd_header_len         
%К"
bwd_iat_max         
&К#
bwd_iat_mean         
%К"
bwd_iat_min         
%К"
bwd_iat_std         
%К"
bwd_iat_tot         
)К&
bwd_pkt_len_max         
*К'
bwd_pkt_len_mean         
)К&
bwd_pkt_len_min         
)К&
bwd_pkt_len_std         
(К%
bwd_pkts_b_avg         
$К!

bwd_pkts_s         
'К$
bwd_psh_flags         
'К$
bwd_urg_flags         
'К$
down_up_ratio         
"К
dst_port         
&К#
ece_flag_cnt         
&К#
fin_flag_cnt         
%К"
flow_byts_s         
'К$
flow_duration         
&К#
flow_iat_max         
'К$
flow_iat_mean         
&К#
flow_iat_min         
&К#
flow_iat_std         
%К"
flow_pkts_s         
+К(
fwd_act_data_pkts         
*К'
fwd_blk_rate_avg         
(К%
fwd_byts_b_avg         
(К%
fwd_header_len         
%К"
fwd_iat_max         
&К#
fwd_iat_mean         
%К"
fwd_iat_min         
%К"
fwd_iat_std         
%К"
fwd_iat_tot         
)К&
fwd_pkt_len_max         
*К'
fwd_pkt_len_mean         
)К&
fwd_pkt_len_min         
)К&
fwd_pkt_len_std         
(К%
fwd_pkts_b_avg         
$К!

fwd_pkts_s         
'К$
fwd_psh_flags         
*К'
fwd_seg_size_min         
'К$
fwd_urg_flags         
"К
idle_max         
#К 
	idle_mean         
"К
idle_min         
"К
idle_std         
К
info         
+К(
init_bwd_win_byts         
+К(
init_fwd_win_byts         
%К"
pkt_len_max         
&К#
pkt_len_mean         
%К"
pkt_len_min         
%К"
pkt_len_std         
%К"
pkt_len_var         
$К!

pkt_length         
&К#
pkt_size_avg         
"К
protocol         
&К#
psh_flag_cnt         
&К#
rst_flag_cnt         
"К
src_port         
&К#
syn_flag_cnt         
&К#
tot_bwd_pkts         
&К#
tot_fwd_pkts         
)К&
totlen_bwd_pkts         
)К&
totlen_fwd_pkts         
&К#
urg_flag_cnt         
к "!К
unknown         ═!
,__inference_signature_wrapper___call___17769Ь!	р в▄ 
в 
╘ к╨ 
6
ack_flag_cnt&К#
ack_flag_cnt         
2

active_max$К!

active_max         
4
active_mean%К"
active_mean         
2

active_min$К!

active_min         
2

active_std$К!

active_std         
>
bwd_blk_rate_avg*К'
bwd_blk_rate_avg         
:
bwd_byts_b_avg(К%
bwd_byts_b_avg         
:
bwd_header_len(К%
bwd_header_len         
4
bwd_iat_max%К"
bwd_iat_max         
6
bwd_iat_mean&К#
bwd_iat_mean         
4
bwd_iat_min%К"
bwd_iat_min         
4
bwd_iat_std%К"
bwd_iat_std         
4
bwd_iat_tot%К"
bwd_iat_tot         
<
bwd_pkt_len_max)К&
bwd_pkt_len_max         
>
bwd_pkt_len_mean*К'
bwd_pkt_len_mean         
<
bwd_pkt_len_min)К&
bwd_pkt_len_min         
<
bwd_pkt_len_std)К&
bwd_pkt_len_std         
:
bwd_pkts_b_avg(К%
bwd_pkts_b_avg         
2

bwd_pkts_s$К!

bwd_pkts_s         
8
bwd_psh_flags'К$
bwd_psh_flags         
8
bwd_urg_flags'К$
bwd_urg_flags         
8
down_up_ratio'К$
down_up_ratio         
.
dst_port"К
dst_port         
6
ece_flag_cnt&К#
ece_flag_cnt         
6
fin_flag_cnt&К#
fin_flag_cnt         
4
flow_byts_s%К"
flow_byts_s         
8
flow_duration'К$
flow_duration         
6
flow_iat_max&К#
flow_iat_max         
8
flow_iat_mean'К$
flow_iat_mean         
6
flow_iat_min&К#
flow_iat_min         
6
flow_iat_std&К#
flow_iat_std         
4
flow_pkts_s%К"
flow_pkts_s         
@
fwd_act_data_pkts+К(
fwd_act_data_pkts         
>
fwd_blk_rate_avg*К'
fwd_blk_rate_avg         
:
fwd_byts_b_avg(К%
fwd_byts_b_avg         
:
fwd_header_len(К%
fwd_header_len         
4
fwd_iat_max%К"
fwd_iat_max         
6
fwd_iat_mean&К#
fwd_iat_mean         
4
fwd_iat_min%К"
fwd_iat_min         
4
fwd_iat_std%К"
fwd_iat_std         
4
fwd_iat_tot%К"
fwd_iat_tot         
<
fwd_pkt_len_max)К&
fwd_pkt_len_max         
>
fwd_pkt_len_mean*К'
fwd_pkt_len_mean         
<
fwd_pkt_len_min)К&
fwd_pkt_len_min         
<
fwd_pkt_len_std)К&
fwd_pkt_len_std         
:
fwd_pkts_b_avg(К%
fwd_pkts_b_avg         
2

fwd_pkts_s$К!

fwd_pkts_s         
8
fwd_psh_flags'К$
fwd_psh_flags         
>
fwd_seg_size_min*К'
fwd_seg_size_min         
8
fwd_urg_flags'К$
fwd_urg_flags         
.
idle_max"К
idle_max         
0
	idle_mean#К 
	idle_mean         
.
idle_min"К
idle_min         
.
idle_std"К
idle_std         
&
infoК
info         
@
init_bwd_win_byts+К(
init_bwd_win_byts         
@
init_fwd_win_byts+К(
init_fwd_win_byts         
4
pkt_len_max%К"
pkt_len_max         
6
pkt_len_mean&К#
pkt_len_mean         
4
pkt_len_min%К"
pkt_len_min         
4
pkt_len_std%К"
pkt_len_std         
4
pkt_len_var%К"
pkt_len_var         
2

pkt_length$К!

pkt_length         
6
pkt_size_avg&К#
pkt_size_avg         
.
protocol"К
protocol         
6
psh_flag_cnt&К#
psh_flag_cnt         
6
rst_flag_cnt&К#
rst_flag_cnt         
.
src_port"К
src_port         
6
syn_flag_cnt&К#
syn_flag_cnt         
6
tot_bwd_pkts&К#
tot_bwd_pkts         
6
tot_fwd_pkts&К#
tot_fwd_pkts         
<
totlen_bwd_pkts)К&
totlen_bwd_pkts         
<
totlen_fwd_pkts)К&
totlen_fwd_pkts         
6
urg_flag_cnt&К#
urg_flag_cnt         "3к0
.
output_0"К
output_0         ═!
,__inference_signature_wrapper___call___17851Ь!	р в▄ 
в 
╘ к╨ 
6
ack_flag_cnt&К#
ack_flag_cnt         
2

active_max$К!

active_max         
4
active_mean%К"
active_mean         
2

active_min$К!

active_min         
2

active_std$К!

active_std         
>
bwd_blk_rate_avg*К'
bwd_blk_rate_avg         
:
bwd_byts_b_avg(К%
bwd_byts_b_avg         
:
bwd_header_len(К%
bwd_header_len         
4
bwd_iat_max%К"
bwd_iat_max         
6
bwd_iat_mean&К#
bwd_iat_mean         
4
bwd_iat_min%К"
bwd_iat_min         
4
bwd_iat_std%К"
bwd_iat_std         
4
bwd_iat_tot%К"
bwd_iat_tot         
<
bwd_pkt_len_max)К&
bwd_pkt_len_max         
>
bwd_pkt_len_mean*К'
bwd_pkt_len_mean         
<
bwd_pkt_len_min)К&
bwd_pkt_len_min         
<
bwd_pkt_len_std)К&
bwd_pkt_len_std         
:
bwd_pkts_b_avg(К%
bwd_pkts_b_avg         
2

bwd_pkts_s$К!

bwd_pkts_s         
8
bwd_psh_flags'К$
bwd_psh_flags         
8
bwd_urg_flags'К$
bwd_urg_flags         
8
down_up_ratio'К$
down_up_ratio         
.
dst_port"К
dst_port         
6
ece_flag_cnt&К#
ece_flag_cnt         
6
fin_flag_cnt&К#
fin_flag_cnt         
4
flow_byts_s%К"
flow_byts_s         
8
flow_duration'К$
flow_duration         
6
flow_iat_max&К#
flow_iat_max         
8
flow_iat_mean'К$
flow_iat_mean         
6
flow_iat_min&К#
flow_iat_min         
6
flow_iat_std&К#
flow_iat_std         
4
flow_pkts_s%К"
flow_pkts_s         
@
fwd_act_data_pkts+К(
fwd_act_data_pkts         
>
fwd_blk_rate_avg*К'
fwd_blk_rate_avg         
:
fwd_byts_b_avg(К%
fwd_byts_b_avg         
:
fwd_header_len(К%
fwd_header_len         
4
fwd_iat_max%К"
fwd_iat_max         
6
fwd_iat_mean&К#
fwd_iat_mean         
4
fwd_iat_min%К"
fwd_iat_min         
4
fwd_iat_std%К"
fwd_iat_std         
4
fwd_iat_tot%К"
fwd_iat_tot         
<
fwd_pkt_len_max)К&
fwd_pkt_len_max         
>
fwd_pkt_len_mean*К'
fwd_pkt_len_mean         
<
fwd_pkt_len_min)К&
fwd_pkt_len_min         
<
fwd_pkt_len_std)К&
fwd_pkt_len_std         
:
fwd_pkts_b_avg(К%
fwd_pkts_b_avg         
2

fwd_pkts_s$К!

fwd_pkts_s         
8
fwd_psh_flags'К$
fwd_psh_flags         
>
fwd_seg_size_min*К'
fwd_seg_size_min         
8
fwd_urg_flags'К$
fwd_urg_flags         
.
idle_max"К
idle_max         
0
	idle_mean#К 
	idle_mean         
.
idle_min"К
idle_min         
.
idle_std"К
idle_std         
&
infoК
info         
@
init_bwd_win_byts+К(
init_bwd_win_byts         
@
init_fwd_win_byts+К(
init_fwd_win_byts         
4
pkt_len_max%К"
pkt_len_max         
6
pkt_len_mean&К#
pkt_len_mean         
4
pkt_len_min%К"
pkt_len_min         
4
pkt_len_std%К"
pkt_len_std         
4
pkt_len_var%К"
pkt_len_var         
2

pkt_length$К!

pkt_length         
6
pkt_size_avg&К#
pkt_size_avg         
.
protocol"К
protocol         
6
psh_flag_cnt&К#
psh_flag_cnt         
6
rst_flag_cnt&К#
rst_flag_cnt         
.
src_port"К
src_port         
6
syn_flag_cnt&К#
syn_flag_cnt         
6
tot_bwd_pkts&К#
tot_bwd_pkts         
6
tot_fwd_pkts&К#
tot_fwd_pkts         
<
totlen_bwd_pkts)К&
totlen_bwd_pkts         
<
totlen_fwd_pkts)К&
totlen_fwd_pkts         
6
urg_flag_cnt&К#
urg_flag_cnt         "3к0
.
output_0"К
output_0         