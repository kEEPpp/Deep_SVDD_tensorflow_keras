Ö
²
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
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
¾
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
executor_typestring 
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8þ

deep_svdd_4/encoder1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ô*,
shared_namedeep_svdd_4/encoder1/kernel

/deep_svdd_4/encoder1/kernel/Read/ReadVariableOpReadVariableOpdeep_svdd_4/encoder1/kernel* 
_output_shapes
:
ô*
dtype0

deep_svdd_4/encoder1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ô**
shared_namedeep_svdd_4/encoder1/bias

-deep_svdd_4/encoder1/bias/Read/ReadVariableOpReadVariableOpdeep_svdd_4/encoder1/bias*
_output_shapes	
:ô*
dtype0

deep_svdd_4/encoder2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ôÞ*,
shared_namedeep_svdd_4/encoder2/kernel

/deep_svdd_4/encoder2/kernel/Read/ReadVariableOpReadVariableOpdeep_svdd_4/encoder2/kernel* 
_output_shapes
:
ôÞ*
dtype0

deep_svdd_4/encoder2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Þ**
shared_namedeep_svdd_4/encoder2/bias

-deep_svdd_4/encoder2/bias/Read/ReadVariableOpReadVariableOpdeep_svdd_4/encoder2/bias*
_output_shapes	
:Þ*
dtype0

deep_svdd_4/latent/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Þ**
shared_namedeep_svdd_4/latent/kernel

-deep_svdd_4/latent/kernel/Read/ReadVariableOpReadVariableOpdeep_svdd_4/latent/kernel*
_output_shapes
:	Þ*
dtype0

deep_svdd_4/latent/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namedeep_svdd_4/latent/bias

+deep_svdd_4/latent/bias/Read/ReadVariableOpReadVariableOpdeep_svdd_4/latent/bias*
_output_shapes
:*
dtype0

NoOpNoOp
¡
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ü
valueÒBÏ BÈ

encoder_layer1
encoder_layer2

latent
trainable_variables
regularization_losses
	variables
	keras_api

signatures
h

	kernel

bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*
	0

1
2
3
4
5
 
*
	0

1
2
3
4
5
­
layer_metrics
metrics
trainable_variables

layers
layer_regularization_losses
regularization_losses
non_trainable_variables
	variables
 
a_
VARIABLE_VALUEdeep_svdd_4/encoder1/kernel0encoder_layer1/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEdeep_svdd_4/encoder1/bias.encoder_layer1/bias/.ATTRIBUTES/VARIABLE_VALUE

	0

1
 

	0

1
­
 layer_metrics
!metrics
trainable_variables

"layers
#layer_regularization_losses
regularization_losses
$non_trainable_variables
	variables
a_
VARIABLE_VALUEdeep_svdd_4/encoder2/kernel0encoder_layer2/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEdeep_svdd_4/encoder2/bias.encoder_layer2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
%layer_metrics
&metrics
trainable_variables

'layers
(layer_regularization_losses
regularization_losses
)non_trainable_variables
	variables
WU
VARIABLE_VALUEdeep_svdd_4/latent/kernel(latent/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdeep_svdd_4/latent/bias&latent/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
*layer_metrics
+metrics
trainable_variables

,layers
-layer_regularization_losses
regularization_losses
.non_trainable_variables
	variables
 
 

0
1
2
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
|
serving_default_input_1Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
à
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1deep_svdd_4/encoder1/kerneldeep_svdd_4/encoder1/biasdeep_svdd_4/encoder2/kerneldeep_svdd_4/encoder2/biasdeep_svdd_4/latent/kerneldeep_svdd_4/latent/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_1677
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
»
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/deep_svdd_4/encoder1/kernel/Read/ReadVariableOp-deep_svdd_4/encoder1/bias/Read/ReadVariableOp/deep_svdd_4/encoder2/kernel/Read/ReadVariableOp-deep_svdd_4/encoder2/bias/Read/ReadVariableOp-deep_svdd_4/latent/kernel/Read/ReadVariableOp+deep_svdd_4/latent/bias/Read/ReadVariableOpConst*
Tin

2*
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
GPU 2J 8 *&
f!R
__inference__traced_save_1946
¾
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedeep_svdd_4/encoder1/kerneldeep_svdd_4/encoder1/biasdeep_svdd_4/encoder2/kerneldeep_svdd_4/encoder2/biasdeep_svdd_4/latent/kerneldeep_svdd_4/latent/bias*
Tin
	2*
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
GPU 2J 8 *)
f$R"
 __inference__traced_restore_1974ñï
Ü
|
'__inference_encoder2_layer_call_fn_1885

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_encoder2_layer_call_and_return_conditional_losses_15222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿô::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
í
¿
*__inference_deep_svdd_4_layer_call_fn_1845

input_data
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCall
input_dataunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_deep_svdd_4_layer_call_and_return_conditional_losses_16432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
input_data
õ	
Û
B__inference_encoder2_layer_call_and_return_conditional_losses_1522

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ôÞ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Þ*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿô::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
ä
¼
*__inference_deep_svdd_4_layer_call_fn_1744
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_deep_svdd_4_layer_call_and_return_conditional_losses_16072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¶
´
"__inference_signature_wrapper_1677
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_14802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ç
î
 __inference__traced_restore_1974
file_prefix0
,assignvariableop_deep_svdd_4_encoder1_kernel0
,assignvariableop_1_deep_svdd_4_encoder1_bias2
.assignvariableop_2_deep_svdd_4_encoder2_kernel0
,assignvariableop_3_deep_svdd_4_encoder2_bias0
,assignvariableop_4_deep_svdd_4_latent_kernel.
*assignvariableop_5_deep_svdd_4_latent_bias

identity_7¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5½
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*É
value¿B¼B0encoder_layer1/kernel/.ATTRIBUTES/VARIABLE_VALUEB.encoder_layer1/bias/.ATTRIBUTES/VARIABLE_VALUEB0encoder_layer2/kernel/.ATTRIBUTES/VARIABLE_VALUEB.encoder_layer2/bias/.ATTRIBUTES/VARIABLE_VALUEB(latent/kernel/.ATTRIBUTES/VARIABLE_VALUEB&latent/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slicesÎ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity«
AssignVariableOpAssignVariableOp,assignvariableop_deep_svdd_4_encoder1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1±
AssignVariableOp_1AssignVariableOp,assignvariableop_1_deep_svdd_4_encoder1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2³
AssignVariableOp_2AssignVariableOp.assignvariableop_2_deep_svdd_4_encoder2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3±
AssignVariableOp_3AssignVariableOp,assignvariableop_3_deep_svdd_4_encoder2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4±
AssignVariableOp_4AssignVariableOp,assignvariableop_4_deep_svdd_4_latent_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¯
AssignVariableOp_5AssignVariableOp*assignvariableop_5_deep_svdd_4_latent_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpä

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6Ö

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ä
¼
*__inference_deep_svdd_4_layer_call_fn_1761
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_deep_svdd_4_layer_call_and_return_conditional_losses_16432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ü
|
'__inference_encoder1_layer_call_fn_1865

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_encoder1_layer_call_and_return_conditional_losses_14952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
¿
*__inference_deep_svdd_4_layer_call_fn_1828

input_data
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCall
input_dataunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_deep_svdd_4_layer_call_and_return_conditional_losses_16072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
input_data
ö$
¡
__inference__wrapped_model_1480
input_17
3deep_svdd_4_encoder1_matmul_readvariableop_resource8
4deep_svdd_4_encoder1_biasadd_readvariableop_resource7
3deep_svdd_4_encoder2_matmul_readvariableop_resource8
4deep_svdd_4_encoder2_biasadd_readvariableop_resource5
1deep_svdd_4_latent_matmul_readvariableop_resource6
2deep_svdd_4_latent_biasadd_readvariableop_resource
identity¢+deep_svdd_4/encoder1/BiasAdd/ReadVariableOp¢*deep_svdd_4/encoder1/MatMul/ReadVariableOp¢+deep_svdd_4/encoder2/BiasAdd/ReadVariableOp¢*deep_svdd_4/encoder2/MatMul/ReadVariableOp¢)deep_svdd_4/latent/BiasAdd/ReadVariableOp¢(deep_svdd_4/latent/MatMul/ReadVariableOpÎ
*deep_svdd_4/encoder1/MatMul/ReadVariableOpReadVariableOp3deep_svdd_4_encoder1_matmul_readvariableop_resource* 
_output_shapes
:
ô*
dtype02,
*deep_svdd_4/encoder1/MatMul/ReadVariableOp´
deep_svdd_4/encoder1/MatMulMatMulinput_12deep_svdd_4/encoder1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
deep_svdd_4/encoder1/MatMulÌ
+deep_svdd_4/encoder1/BiasAdd/ReadVariableOpReadVariableOp4deep_svdd_4_encoder1_biasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype02-
+deep_svdd_4/encoder1/BiasAdd/ReadVariableOpÖ
deep_svdd_4/encoder1/BiasAddBiasAdd%deep_svdd_4/encoder1/MatMul:product:03deep_svdd_4/encoder1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
deep_svdd_4/encoder1/BiasAdd
deep_svdd_4/encoder1/ReluRelu%deep_svdd_4/encoder1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
deep_svdd_4/encoder1/ReluÎ
*deep_svdd_4/encoder2/MatMul/ReadVariableOpReadVariableOp3deep_svdd_4_encoder2_matmul_readvariableop_resource* 
_output_shapes
:
ôÞ*
dtype02,
*deep_svdd_4/encoder2/MatMul/ReadVariableOpÔ
deep_svdd_4/encoder2/MatMulMatMul'deep_svdd_4/encoder1/Relu:activations:02deep_svdd_4/encoder2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2
deep_svdd_4/encoder2/MatMulÌ
+deep_svdd_4/encoder2/BiasAdd/ReadVariableOpReadVariableOp4deep_svdd_4_encoder2_biasadd_readvariableop_resource*
_output_shapes	
:Þ*
dtype02-
+deep_svdd_4/encoder2/BiasAdd/ReadVariableOpÖ
deep_svdd_4/encoder2/BiasAddBiasAdd%deep_svdd_4/encoder2/MatMul:product:03deep_svdd_4/encoder2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2
deep_svdd_4/encoder2/BiasAdd
deep_svdd_4/encoder2/ReluRelu%deep_svdd_4/encoder2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2
deep_svdd_4/encoder2/ReluÇ
(deep_svdd_4/latent/MatMul/ReadVariableOpReadVariableOp1deep_svdd_4_latent_matmul_readvariableop_resource*
_output_shapes
:	Þ*
dtype02*
(deep_svdd_4/latent/MatMul/ReadVariableOpÍ
deep_svdd_4/latent/MatMulMatMul'deep_svdd_4/encoder2/Relu:activations:00deep_svdd_4/latent/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
deep_svdd_4/latent/MatMulÅ
)deep_svdd_4/latent/BiasAdd/ReadVariableOpReadVariableOp2deep_svdd_4_latent_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)deep_svdd_4/latent/BiasAdd/ReadVariableOpÍ
deep_svdd_4/latent/BiasAddBiasAdd#deep_svdd_4/latent/MatMul:product:01deep_svdd_4/latent/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
deep_svdd_4/latent/BiasAdd
deep_svdd_4/latent/ReluRelu#deep_svdd_4/latent/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
deep_svdd_4/latent/Relu
IdentityIdentity%deep_svdd_4/latent/Relu:activations:0,^deep_svdd_4/encoder1/BiasAdd/ReadVariableOp+^deep_svdd_4/encoder1/MatMul/ReadVariableOp,^deep_svdd_4/encoder2/BiasAdd/ReadVariableOp+^deep_svdd_4/encoder2/MatMul/ReadVariableOp*^deep_svdd_4/latent/BiasAdd/ReadVariableOp)^deep_svdd_4/latent/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::::2Z
+deep_svdd_4/encoder1/BiasAdd/ReadVariableOp+deep_svdd_4/encoder1/BiasAdd/ReadVariableOp2X
*deep_svdd_4/encoder1/MatMul/ReadVariableOp*deep_svdd_4/encoder1/MatMul/ReadVariableOp2Z
+deep_svdd_4/encoder2/BiasAdd/ReadVariableOp+deep_svdd_4/encoder2/BiasAdd/ReadVariableOp2X
*deep_svdd_4/encoder2/MatMul/ReadVariableOp*deep_svdd_4/encoder2/MatMul/ReadVariableOp2V
)deep_svdd_4/latent/BiasAdd/ReadVariableOp)deep_svdd_4/latent/BiasAdd/ReadVariableOp2T
(deep_svdd_4/latent/MatMul/ReadVariableOp(deep_svdd_4/latent/MatMul/ReadVariableOp:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ö
º
E__inference_deep_svdd_4_layer_call_and_return_conditional_losses_1811

input_data+
'encoder1_matmul_readvariableop_resource,
(encoder1_biasadd_readvariableop_resource+
'encoder2_matmul_readvariableop_resource,
(encoder2_biasadd_readvariableop_resource)
%latent_matmul_readvariableop_resource*
&latent_biasadd_readvariableop_resource
identity¢encoder1/BiasAdd/ReadVariableOp¢encoder1/MatMul/ReadVariableOp¢encoder2/BiasAdd/ReadVariableOp¢encoder2/MatMul/ReadVariableOp¢latent/BiasAdd/ReadVariableOp¢latent/MatMul/ReadVariableOpª
encoder1/MatMul/ReadVariableOpReadVariableOp'encoder1_matmul_readvariableop_resource* 
_output_shapes
:
ô*
dtype02 
encoder1/MatMul/ReadVariableOp
encoder1/MatMulMatMul
input_data&encoder1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
encoder1/MatMul¨
encoder1/BiasAdd/ReadVariableOpReadVariableOp(encoder1_biasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype02!
encoder1/BiasAdd/ReadVariableOp¦
encoder1/BiasAddBiasAddencoder1/MatMul:product:0'encoder1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
encoder1/BiasAddt
encoder1/ReluReluencoder1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
encoder1/Reluª
encoder2/MatMul/ReadVariableOpReadVariableOp'encoder2_matmul_readvariableop_resource* 
_output_shapes
:
ôÞ*
dtype02 
encoder2/MatMul/ReadVariableOp¤
encoder2/MatMulMatMulencoder1/Relu:activations:0&encoder2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2
encoder2/MatMul¨
encoder2/BiasAdd/ReadVariableOpReadVariableOp(encoder2_biasadd_readvariableop_resource*
_output_shapes	
:Þ*
dtype02!
encoder2/BiasAdd/ReadVariableOp¦
encoder2/BiasAddBiasAddencoder2/MatMul:product:0'encoder2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2
encoder2/BiasAddt
encoder2/ReluReluencoder2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2
encoder2/Relu£
latent/MatMul/ReadVariableOpReadVariableOp%latent_matmul_readvariableop_resource*
_output_shapes
:	Þ*
dtype02
latent/MatMul/ReadVariableOp
latent/MatMulMatMulencoder2/Relu:activations:0$latent/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
latent/MatMul¡
latent/BiasAdd/ReadVariableOpReadVariableOp&latent_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
latent/BiasAdd/ReadVariableOp
latent/BiasAddBiasAddlatent/MatMul:product:0%latent/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
latent/BiasAddm
latent/ReluRelulatent/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
latent/Relu²
IdentityIdentitylatent/Relu:activations:0 ^encoder1/BiasAdd/ReadVariableOp^encoder1/MatMul/ReadVariableOp ^encoder2/BiasAdd/ReadVariableOp^encoder2/MatMul/ReadVariableOp^latent/BiasAdd/ReadVariableOp^latent/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::::2B
encoder1/BiasAdd/ReadVariableOpencoder1/BiasAdd/ReadVariableOp2@
encoder1/MatMul/ReadVariableOpencoder1/MatMul/ReadVariableOp2B
encoder2/BiasAdd/ReadVariableOpencoder2/BiasAdd/ReadVariableOp2@
encoder2/MatMul/ReadVariableOpencoder2/MatMul/ReadVariableOp2>
latent/BiasAdd/ReadVariableOplatent/BiasAdd/ReadVariableOp2<
latent/MatMul/ReadVariableOplatent/MatMul/ReadVariableOp:T P
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
input_data
õ	
Û
B__inference_encoder2_layer_call_and_return_conditional_losses_1876

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ôÞ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Þ*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿô::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
í	
Ù
@__inference_latent_layer_call_and_return_conditional_losses_1896

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Þ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÞ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ
 
_user_specified_nameinputs
Í
·
E__inference_deep_svdd_4_layer_call_and_return_conditional_losses_1702
input_1+
'encoder1_matmul_readvariableop_resource,
(encoder1_biasadd_readvariableop_resource+
'encoder2_matmul_readvariableop_resource,
(encoder2_biasadd_readvariableop_resource)
%latent_matmul_readvariableop_resource*
&latent_biasadd_readvariableop_resource
identity¢encoder1/BiasAdd/ReadVariableOp¢encoder1/MatMul/ReadVariableOp¢encoder2/BiasAdd/ReadVariableOp¢encoder2/MatMul/ReadVariableOp¢latent/BiasAdd/ReadVariableOp¢latent/MatMul/ReadVariableOpª
encoder1/MatMul/ReadVariableOpReadVariableOp'encoder1_matmul_readvariableop_resource* 
_output_shapes
:
ô*
dtype02 
encoder1/MatMul/ReadVariableOp
encoder1/MatMulMatMulinput_1&encoder1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
encoder1/MatMul¨
encoder1/BiasAdd/ReadVariableOpReadVariableOp(encoder1_biasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype02!
encoder1/BiasAdd/ReadVariableOp¦
encoder1/BiasAddBiasAddencoder1/MatMul:product:0'encoder1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
encoder1/BiasAddt
encoder1/ReluReluencoder1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
encoder1/Reluª
encoder2/MatMul/ReadVariableOpReadVariableOp'encoder2_matmul_readvariableop_resource* 
_output_shapes
:
ôÞ*
dtype02 
encoder2/MatMul/ReadVariableOp¤
encoder2/MatMulMatMulencoder1/Relu:activations:0&encoder2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2
encoder2/MatMul¨
encoder2/BiasAdd/ReadVariableOpReadVariableOp(encoder2_biasadd_readvariableop_resource*
_output_shapes	
:Þ*
dtype02!
encoder2/BiasAdd/ReadVariableOp¦
encoder2/BiasAddBiasAddencoder2/MatMul:product:0'encoder2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2
encoder2/BiasAddt
encoder2/ReluReluencoder2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2
encoder2/Relu£
latent/MatMul/ReadVariableOpReadVariableOp%latent_matmul_readvariableop_resource*
_output_shapes
:	Þ*
dtype02
latent/MatMul/ReadVariableOp
latent/MatMulMatMulencoder2/Relu:activations:0$latent/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
latent/MatMul¡
latent/BiasAdd/ReadVariableOpReadVariableOp&latent_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
latent/BiasAdd/ReadVariableOp
latent/BiasAddBiasAddlatent/MatMul:product:0%latent/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
latent/BiasAddm
latent/ReluRelulatent/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
latent/Relu²
IdentityIdentitylatent/Relu:activations:0 ^encoder1/BiasAdd/ReadVariableOp^encoder1/MatMul/ReadVariableOp ^encoder2/BiasAdd/ReadVariableOp^encoder2/MatMul/ReadVariableOp^latent/BiasAdd/ReadVariableOp^latent/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::::2B
encoder1/BiasAdd/ReadVariableOpencoder1/BiasAdd/ReadVariableOp2@
encoder1/MatMul/ReadVariableOpencoder1/MatMul/ReadVariableOp2B
encoder2/BiasAdd/ReadVariableOpencoder2/BiasAdd/ReadVariableOp2@
encoder2/MatMul/ReadVariableOpencoder2/MatMul/ReadVariableOp2>
latent/BiasAdd/ReadVariableOplatent/BiasAdd/ReadVariableOp2<
latent/MatMul/ReadVariableOplatent/MatMul/ReadVariableOp:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
õ	
Û
B__inference_encoder1_layer_call_and_return_conditional_losses_1495

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ô*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í	
Ù
@__inference_latent_layer_call_and_return_conditional_losses_1549

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Þ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÞ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ
 
_user_specified_nameinputs
Í
·
E__inference_deep_svdd_4_layer_call_and_return_conditional_losses_1727
input_1+
'encoder1_matmul_readvariableop_resource,
(encoder1_biasadd_readvariableop_resource+
'encoder2_matmul_readvariableop_resource,
(encoder2_biasadd_readvariableop_resource)
%latent_matmul_readvariableop_resource*
&latent_biasadd_readvariableop_resource
identity¢encoder1/BiasAdd/ReadVariableOp¢encoder1/MatMul/ReadVariableOp¢encoder2/BiasAdd/ReadVariableOp¢encoder2/MatMul/ReadVariableOp¢latent/BiasAdd/ReadVariableOp¢latent/MatMul/ReadVariableOpª
encoder1/MatMul/ReadVariableOpReadVariableOp'encoder1_matmul_readvariableop_resource* 
_output_shapes
:
ô*
dtype02 
encoder1/MatMul/ReadVariableOp
encoder1/MatMulMatMulinput_1&encoder1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
encoder1/MatMul¨
encoder1/BiasAdd/ReadVariableOpReadVariableOp(encoder1_biasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype02!
encoder1/BiasAdd/ReadVariableOp¦
encoder1/BiasAddBiasAddencoder1/MatMul:product:0'encoder1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
encoder1/BiasAddt
encoder1/ReluReluencoder1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
encoder1/Reluª
encoder2/MatMul/ReadVariableOpReadVariableOp'encoder2_matmul_readvariableop_resource* 
_output_shapes
:
ôÞ*
dtype02 
encoder2/MatMul/ReadVariableOp¤
encoder2/MatMulMatMulencoder1/Relu:activations:0&encoder2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2
encoder2/MatMul¨
encoder2/BiasAdd/ReadVariableOpReadVariableOp(encoder2_biasadd_readvariableop_resource*
_output_shapes	
:Þ*
dtype02!
encoder2/BiasAdd/ReadVariableOp¦
encoder2/BiasAddBiasAddencoder2/MatMul:product:0'encoder2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2
encoder2/BiasAddt
encoder2/ReluReluencoder2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2
encoder2/Relu£
latent/MatMul/ReadVariableOpReadVariableOp%latent_matmul_readvariableop_resource*
_output_shapes
:	Þ*
dtype02
latent/MatMul/ReadVariableOp
latent/MatMulMatMulencoder2/Relu:activations:0$latent/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
latent/MatMul¡
latent/BiasAdd/ReadVariableOpReadVariableOp&latent_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
latent/BiasAdd/ReadVariableOp
latent/BiasAddBiasAddlatent/MatMul:product:0%latent/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
latent/BiasAddm
latent/ReluRelulatent/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
latent/Relu²
IdentityIdentitylatent/Relu:activations:0 ^encoder1/BiasAdd/ReadVariableOp^encoder1/MatMul/ReadVariableOp ^encoder2/BiasAdd/ReadVariableOp^encoder2/MatMul/ReadVariableOp^latent/BiasAdd/ReadVariableOp^latent/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::::2B
encoder1/BiasAdd/ReadVariableOpencoder1/BiasAdd/ReadVariableOp2@
encoder1/MatMul/ReadVariableOpencoder1/MatMul/ReadVariableOp2B
encoder2/BiasAdd/ReadVariableOpencoder2/BiasAdd/ReadVariableOp2@
encoder2/MatMul/ReadVariableOpencoder2/MatMul/ReadVariableOp2>
latent/BiasAdd/ReadVariableOplatent/BiasAdd/ReadVariableOp2<
latent/MatMul/ReadVariableOplatent/MatMul/ReadVariableOp:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
õ	
Û
B__inference_encoder1_layer_call_and_return_conditional_losses_1856

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ô*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

½
E__inference_deep_svdd_4_layer_call_and_return_conditional_losses_1607

input_data
encoder1_1591
encoder1_1593
encoder2_1596
encoder2_1598
latent_1601
latent_1603
identity¢ encoder1/StatefulPartitionedCall¢ encoder2/StatefulPartitionedCall¢latent/StatefulPartitionedCall
 encoder1/StatefulPartitionedCallStatefulPartitionedCall
input_dataencoder1_1591encoder1_1593*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_encoder1_layer_call_and_return_conditional_losses_14952"
 encoder1/StatefulPartitionedCall²
 encoder2/StatefulPartitionedCallStatefulPartitionedCall)encoder1/StatefulPartitionedCall:output:0encoder2_1596encoder2_1598*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_encoder2_layer_call_and_return_conditional_losses_15222"
 encoder2/StatefulPartitionedCall§
latent/StatefulPartitionedCallStatefulPartitionedCall)encoder2/StatefulPartitionedCall:output:0latent_1601latent_1603*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_latent_layer_call_and_return_conditional_losses_15492 
latent/StatefulPartitionedCallâ
IdentityIdentity'latent/StatefulPartitionedCall:output:0!^encoder1/StatefulPartitionedCall!^encoder2/StatefulPartitionedCall^latent/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::::2D
 encoder1/StatefulPartitionedCall encoder1/StatefulPartitionedCall2D
 encoder2/StatefulPartitionedCall encoder2/StatefulPartitionedCall2@
latent/StatefulPartitionedCalllatent/StatefulPartitionedCall:T P
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
input_data

½
E__inference_deep_svdd_4_layer_call_and_return_conditional_losses_1643

input_data
encoder1_1627
encoder1_1629
encoder2_1632
encoder2_1634
latent_1637
latent_1639
identity¢ encoder1/StatefulPartitionedCall¢ encoder2/StatefulPartitionedCall¢latent/StatefulPartitionedCall
 encoder1/StatefulPartitionedCallStatefulPartitionedCall
input_dataencoder1_1627encoder1_1629*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_encoder1_layer_call_and_return_conditional_losses_14952"
 encoder1/StatefulPartitionedCall²
 encoder2/StatefulPartitionedCallStatefulPartitionedCall)encoder1/StatefulPartitionedCall:output:0encoder2_1632encoder2_1634*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_encoder2_layer_call_and_return_conditional_losses_15222"
 encoder2/StatefulPartitionedCall§
latent/StatefulPartitionedCallStatefulPartitionedCall)encoder2/StatefulPartitionedCall:output:0latent_1637latent_1639*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_latent_layer_call_and_return_conditional_losses_15492 
latent/StatefulPartitionedCallâ
IdentityIdentity'latent/StatefulPartitionedCall:output:0!^encoder1/StatefulPartitionedCall!^encoder2/StatefulPartitionedCall^latent/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::::2D
 encoder1/StatefulPartitionedCall encoder1/StatefulPartitionedCall2D
 encoder2/StatefulPartitionedCall encoder2/StatefulPartitionedCall2@
latent/StatefulPartitionedCalllatent/StatefulPartitionedCall:T P
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
input_data
Ö
º
E__inference_deep_svdd_4_layer_call_and_return_conditional_losses_1786

input_data+
'encoder1_matmul_readvariableop_resource,
(encoder1_biasadd_readvariableop_resource+
'encoder2_matmul_readvariableop_resource,
(encoder2_biasadd_readvariableop_resource)
%latent_matmul_readvariableop_resource*
&latent_biasadd_readvariableop_resource
identity¢encoder1/BiasAdd/ReadVariableOp¢encoder1/MatMul/ReadVariableOp¢encoder2/BiasAdd/ReadVariableOp¢encoder2/MatMul/ReadVariableOp¢latent/BiasAdd/ReadVariableOp¢latent/MatMul/ReadVariableOpª
encoder1/MatMul/ReadVariableOpReadVariableOp'encoder1_matmul_readvariableop_resource* 
_output_shapes
:
ô*
dtype02 
encoder1/MatMul/ReadVariableOp
encoder1/MatMulMatMul
input_data&encoder1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
encoder1/MatMul¨
encoder1/BiasAdd/ReadVariableOpReadVariableOp(encoder1_biasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype02!
encoder1/BiasAdd/ReadVariableOp¦
encoder1/BiasAddBiasAddencoder1/MatMul:product:0'encoder1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
encoder1/BiasAddt
encoder1/ReluReluencoder1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
encoder1/Reluª
encoder2/MatMul/ReadVariableOpReadVariableOp'encoder2_matmul_readvariableop_resource* 
_output_shapes
:
ôÞ*
dtype02 
encoder2/MatMul/ReadVariableOp¤
encoder2/MatMulMatMulencoder1/Relu:activations:0&encoder2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2
encoder2/MatMul¨
encoder2/BiasAdd/ReadVariableOpReadVariableOp(encoder2_biasadd_readvariableop_resource*
_output_shapes	
:Þ*
dtype02!
encoder2/BiasAdd/ReadVariableOp¦
encoder2/BiasAddBiasAddencoder2/MatMul:product:0'encoder2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2
encoder2/BiasAddt
encoder2/ReluReluencoder2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ2
encoder2/Relu£
latent/MatMul/ReadVariableOpReadVariableOp%latent_matmul_readvariableop_resource*
_output_shapes
:	Þ*
dtype02
latent/MatMul/ReadVariableOp
latent/MatMulMatMulencoder2/Relu:activations:0$latent/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
latent/MatMul¡
latent/BiasAdd/ReadVariableOpReadVariableOp&latent_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
latent/BiasAdd/ReadVariableOp
latent/BiasAddBiasAddlatent/MatMul:product:0%latent/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
latent/BiasAddm
latent/ReluRelulatent/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
latent/Relu²
IdentityIdentitylatent/Relu:activations:0 ^encoder1/BiasAdd/ReadVariableOp^encoder1/MatMul/ReadVariableOp ^encoder2/BiasAdd/ReadVariableOp^encoder2/MatMul/ReadVariableOp^latent/BiasAdd/ReadVariableOp^latent/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::::2B
encoder1/BiasAdd/ReadVariableOpencoder1/BiasAdd/ReadVariableOp2@
encoder1/MatMul/ReadVariableOpencoder1/MatMul/ReadVariableOp2B
encoder2/BiasAdd/ReadVariableOpencoder2/BiasAdd/ReadVariableOp2@
encoder2/MatMul/ReadVariableOpencoder2/MatMul/ReadVariableOp2>
latent/BiasAdd/ReadVariableOplatent/BiasAdd/ReadVariableOp2<
latent/MatMul/ReadVariableOplatent/MatMul/ReadVariableOp:T P
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
input_data
¹
È
__inference__traced_save_1946
file_prefix:
6savev2_deep_svdd_4_encoder1_kernel_read_readvariableop8
4savev2_deep_svdd_4_encoder1_bias_read_readvariableop:
6savev2_deep_svdd_4_encoder2_kernel_read_readvariableop8
4savev2_deep_svdd_4_encoder2_bias_read_readvariableop8
4savev2_deep_svdd_4_latent_kernel_read_readvariableop6
2savev2_deep_svdd_4_latent_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename·
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*É
value¿B¼B0encoder_layer1/kernel/.ATTRIBUTES/VARIABLE_VALUEB.encoder_layer1/bias/.ATTRIBUTES/VARIABLE_VALUEB0encoder_layer2/kernel/.ATTRIBUTES/VARIABLE_VALUEB.encoder_layer2/bias/.ATTRIBUTES/VARIABLE_VALUEB(latent/kernel/.ATTRIBUTES/VARIABLE_VALUEB&latent/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_deep_svdd_4_encoder1_kernel_read_readvariableop4savev2_deep_svdd_4_encoder1_bias_read_readvariableop6savev2_deep_svdd_4_encoder2_kernel_read_readvariableop4savev2_deep_svdd_4_encoder2_bias_read_readvariableop4savev2_deep_svdd_4_latent_kernel_read_readvariableop2savev2_deep_svdd_4_latent_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*N
_input_shapes=
;: :
ô:ô:
ôÞ:Þ:	Þ:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
ô:!

_output_shapes	
:ô:&"
 
_output_shapes
:
ôÞ:!

_output_shapes	
:Þ:%!

_output_shapes
:	Þ: 

_output_shapes
::

_output_shapes
: 
Ö
z
%__inference_latent_layer_call_fn_1905

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_latent_layer_call_and_return_conditional_losses_15492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÞ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¬
serving_default
<
input_11
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¾[
ì
encoder_layer1
encoder_layer2

latent
trainable_variables
regularization_losses
	variables
	keras_api

signatures
/__call__
0_default_save_signature
*1&call_and_return_all_conditional_losses"ü
_tf_keras_modelâ{"class_name": "DeepSVDD", "name": "deep_svdd_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "DeepSVDD"}}
ò

	kernel

bias
trainable_variables
regularization_losses
	variables
	keras_api
2__call__
*3&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Dense", "name": "encoder1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder1", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 784]}}
ò

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
4__call__
*5&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Dense", "name": "encoder2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "encoder2", "trainable": true, "dtype": "float32", "units": 350, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 500}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 500]}}
ì

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
6__call__
*7&call_and_return_all_conditional_losses"Ç
_tf_keras_layer­{"class_name": "Dense", "name": "latent", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "latent", "trainable": true, "dtype": "float32", "units": 2, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 350}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 350]}}
J
	0

1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
	0

1
2
3
4
5"
trackable_list_wrapper
Ê
layer_metrics
metrics
trainable_variables

layers
layer_regularization_losses
regularization_losses
non_trainable_variables
	variables
/__call__
0_default_save_signature
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
,
8serving_default"
signature_map
/:-
ô2deep_svdd_4/encoder1/kernel
(:&ô2deep_svdd_4/encoder1/bias
.
	0

1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
­
 layer_metrics
!metrics
trainable_variables

"layers
#layer_regularization_losses
regularization_losses
$non_trainable_variables
	variables
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
/:-
ôÞ2deep_svdd_4/encoder2/kernel
(:&Þ2deep_svdd_4/encoder2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
%layer_metrics
&metrics
trainable_variables

'layers
(layer_regularization_losses
regularization_losses
)non_trainable_variables
	variables
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
,:*	Þ2deep_svdd_4/latent/kernel
%:#2deep_svdd_4/latent/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
*layer_metrics
+metrics
trainable_variables

,layers
-layer_regularization_losses
regularization_losses
.non_trainable_variables
	variables
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
0
1
2"
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
2ý
*__inference_deep_svdd_4_layer_call_fn_1744
*__inference_deep_svdd_4_layer_call_fn_1761
*__inference_deep_svdd_4_layer_call_fn_1845
*__inference_deep_svdd_4_layer_call_fn_1828Ê
Á²½
FullArgSpec!
args
jself
j
input_data
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Þ2Û
__inference__wrapped_model_1480·
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *'¢$
"
input_1ÿÿÿÿÿÿÿÿÿ
ì2é
E__inference_deep_svdd_4_layer_call_and_return_conditional_losses_1786
E__inference_deep_svdd_4_layer_call_and_return_conditional_losses_1702
E__inference_deep_svdd_4_layer_call_and_return_conditional_losses_1811
E__inference_deep_svdd_4_layer_call_and_return_conditional_losses_1727Ê
Á²½
FullArgSpec!
args
jself
j
input_data
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ñ2Î
'__inference_encoder1_layer_call_fn_1865¢
²
FullArgSpec
args
jself
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
annotationsª *
 
ì2é
B__inference_encoder1_layer_call_and_return_conditional_losses_1856¢
²
FullArgSpec
args
jself
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
annotationsª *
 
Ñ2Î
'__inference_encoder2_layer_call_fn_1885¢
²
FullArgSpec
args
jself
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
annotationsª *
 
ì2é
B__inference_encoder2_layer_call_and_return_conditional_losses_1876¢
²
FullArgSpec
args
jself
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
annotationsª *
 
Ï2Ì
%__inference_latent_layer_call_fn_1905¢
²
FullArgSpec
args
jself
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
annotationsª *
 
ê2ç
@__inference_latent_layer_call_and_return_conditional_losses_1896¢
²
FullArgSpec
args
jself
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
annotationsª *
 
ÉBÆ
"__inference_signature_wrapper_1677input_1"
²
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
annotationsª *
 
__inference__wrapped_model_1480p	
1¢.
'¢$
"
input_1ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ»
E__inference_deep_svdd_4_layer_call_and_return_conditional_losses_1702r	
A¢>
'¢$
"
input_1ÿÿÿÿÿÿÿÿÿ
ª

trainingp"%¢"

0ÿÿÿÿÿÿÿÿÿ
 »
E__inference_deep_svdd_4_layer_call_and_return_conditional_losses_1727r	
A¢>
'¢$
"
input_1ÿÿÿÿÿÿÿÿÿ
ª

trainingp "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
E__inference_deep_svdd_4_layer_call_and_return_conditional_losses_1786u	
D¢A
*¢'
%"

input_dataÿÿÿÿÿÿÿÿÿ
ª

trainingp"%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
E__inference_deep_svdd_4_layer_call_and_return_conditional_losses_1811u	
D¢A
*¢'
%"

input_dataÿÿÿÿÿÿÿÿÿ
ª

trainingp "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_deep_svdd_4_layer_call_fn_1744e	
A¢>
'¢$
"
input_1ÿÿÿÿÿÿÿÿÿ
ª

trainingp"ÿÿÿÿÿÿÿÿÿ
*__inference_deep_svdd_4_layer_call_fn_1761e	
A¢>
'¢$
"
input_1ÿÿÿÿÿÿÿÿÿ
ª

trainingp "ÿÿÿÿÿÿÿÿÿ
*__inference_deep_svdd_4_layer_call_fn_1828h	
D¢A
*¢'
%"

input_dataÿÿÿÿÿÿÿÿÿ
ª

trainingp"ÿÿÿÿÿÿÿÿÿ
*__inference_deep_svdd_4_layer_call_fn_1845h	
D¢A
*¢'
%"

input_dataÿÿÿÿÿÿÿÿÿ
ª

trainingp "ÿÿÿÿÿÿÿÿÿ¤
B__inference_encoder1_layer_call_and_return_conditional_losses_1856^	
0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿô
 |
'__inference_encoder1_layer_call_fn_1865Q	
0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿô¤
B__inference_encoder2_layer_call_and_return_conditional_losses_1876^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿô
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÞ
 |
'__inference_encoder2_layer_call_fn_1885Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿô
ª "ÿÿÿÿÿÿÿÿÿÞ¡
@__inference_latent_layer_call_and_return_conditional_losses_1896]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÞ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
%__inference_latent_layer_call_fn_1905P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÞ
ª "ÿÿÿÿÿÿÿÿÿ¡
"__inference_signature_wrapper_1677{	
<¢9
¢ 
2ª/
-
input_1"
input_1ÿÿÿÿÿÿÿÿÿ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ