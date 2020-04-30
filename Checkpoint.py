"""Utilities for saving/loading Trackable objects."""
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import os
import weakref

import six

import tensorflow as tf

from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops as io_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import saver as v1_saver_lib
from tensorflow.python.training.saving import functional_saver
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.training.tracking import base
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.training.tracking import graph_view as graph_view_lib
from tensorflow.python.training.tracking import tracking
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import lazy_loader
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export

# Loaded lazily due to a circular dependency.
keras_backend = lazy_loader.LazyLoader(
    "keras_backend", globals(),
    "tensorflow.python.keras.backend"
    )


def get_session():
    # Prefer TF's default session since get_session from Keras has side-effects.
    session = ops.get_default_session()
    if session is None:
        session = keras_backend.get_session()
    return session

class Checkpoint(tf.train.Checkpoint):
    def save(self, file_prefix):
        """Saves a training checkpoint and provides basic checkpoint management.
        The saved checkpoint includes variables created by this object and any
        trackable objects it depends on at the time `Checkpoint.save()` is
        called.
        `save` is a basic convenience wrapper around the `write` method,
        sequentially numbering checkpoints using `save_counter` and updating the
        metadata used by `tf.train.latest_checkpoint`. More advanced checkpoint
        management, for example garbage collection and custom numbering, may be
        provided by other utilities which also wrap `write`
        (`tf.train.CheckpointManager` for example).
        Args:
        file_prefix: A prefix to use for the checkpoint filenames
                (/path/to/directory/and_a_prefix). Names are generated based on this
                prefix and `Checkpoint.save_counter`.
        Returns:
        The full path to the checkpoint.
        """
        graph_building = not context.executing_eagerly()
        if graph_building:
            if ops.inside_function():
                raise NotImplementedError(
                    "Calling tf.train.Checkpoint.save() from a function is not "
                    "supported, as save() modifies saving metadata in ways not "
                    "supported by TensorFlow Operations. Consider using "
                    "tf.train.Checkpoint.write(), a lower-level API which does not "
                    "update metadata. tf.train.latest_checkpoint and related APIs will "
                    "not see this checkpoint."
                    )
            session = get_session()
            if self._save_counter is None:
                # When graph building, if this is a new save counter variable then it
                # needs to be initialized before assign_add. This is only an issue if
                # restore() has not been called first.
                session.run(self.save_counter.initializer)
        if not graph_building or self._save_assign_op is None:
            with ops.colocate_with(self.save_counter):
                assign_op = self.save_counter.assign_add(1, read_value=True)
            if graph_building:
                self._save_assign_op = data_structures.NoDependency(assign_op)        
        file_path = self.write(file_prefix)
        checkpoint_management.update_checkpoint_state_internal(
            save_dir=os.path.dirname(file_prefix),
            model_checkpoint_path=file_path,
            all_model_checkpoint_paths=[file_path],
            save_relative_paths=True
            )
        return file_path