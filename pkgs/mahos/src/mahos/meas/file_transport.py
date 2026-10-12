#!/usr/bin/env python3

"""Node and NodeClient integration for shared-file transport.

.. This file is a part of MAHOS project, which is released under the 3-Clause BSD license.
.. See included LICENSE file or https://github.com/ToyotaCRDL/mahos/blob/main/LICENSE for details.

"""

from __future__ import annotations

import os
import typing as T

from mahos.msgs.common_msgs import CleanupArtifactReq, FileArtifact, FileArtifactBundle, Reply
from mahos.util.file_transport import SharedFileTransport


class FileTransportClientMixin(object):
    """Provide shared-file artifact exchange for request clients."""

    def init_file_transport(self, file_transport_dir: str | None):
        """Initialize optional client-side shared-file transport."""

        self.file_transport = (
            SharedFileTransport(file_transport_dir) if file_transport_dir else None
        )

    def request_output(self, legacy_req, artifact_req, destination: str) -> bool:
        """Issue a legacy request or receive a node-created artifact."""

        if self.file_transport is None:
            return self.req.request(legacy_req).success
        return self._receive_artifact(self.req.request(artifact_req), destination)

    def request_input(
        self,
        legacy_req,
        artifact_req_factory: T.Callable[[FileArtifact], T.Any],
        source: str,
    ) -> Reply:
        """Issue a legacy request or stage a local file for node-side loading."""

        if self.file_transport is None:
            return self.req.request(legacy_req)
        try:
            basename = self.file_transport.stage_copy(source, "load")
            path = self.file_transport.resolve(basename)
            artifact = FileArtifact(basename, os.path.getsize(path))
            return self.req.request(artifact_req_factory(artifact))
        except (OSError, ValueError) as exc:
            return Reply(False, str(exc))
        finally:
            if "basename" in locals():
                self.file_transport.remove(basename)

    def _receive_artifact(self, rep: Reply, destination: str) -> bool:
        """Validate and copy returned artifacts, then acknowledge them."""

        if not rep.success or not isinstance(rep.ret, (FileArtifact, FileArtifactBundle)):
            return False
        output = rep.ret
        try:
            if isinstance(output, FileArtifact):
                destinations = [(output, destination)]
            else:
                destinations = [(output.primary, destination)]
                directory = os.path.dirname(destination)
                for file_name, artifact in output.sidecars:
                    file_name = self.file_transport.validate_basename(file_name)
                    destinations.append((artifact, os.path.join(directory, file_name)))

            paths = set()
            for artifact, output_path in destinations:
                if not isinstance(artifact, FileArtifact):
                    return False
                path = os.path.abspath(os.path.expanduser(output_path))
                if path in paths:
                    return False
                paths.add(path)
                source = self.file_transport.resolve(artifact.basename)
                if os.path.getsize(source) != artifact.size:
                    return False
            for artifact, output_path in destinations:
                self.file_transport.copy_from(artifact.basename, output_path)
            return True
        except (OSError, TypeError, ValueError):
            return False
        finally:
            try:
                self.req.request(CleanupArtifactReq(output))
            except Exception:
                pass


class FileTransportNodeMixin(object):
    """Provide shared-file artifact operations for measurement-like nodes.

    The inheriting node must provide ``conf`` and ``logger`` attributes. Call
    :meth:`init_file_transport` after initializing the base Node.

    """

    DEFAULT_TTL = 24.0 * 60.0 * 60.0

    def init_file_transport(
        self,
        output_purposes: T.Iterable[str] = ("save", "export"),
        ttl: float = DEFAULT_TTL,
    ):
        """Initialize transport and clean expired output and load artifacts."""

        purposes = tuple(output_purposes)
        if not purposes or any(purpose not in ("save", "export") for purpose in purposes):
            raise ValueError(f"invalid output artifact purposes: {purposes!r}")
        self._file_transport_output_purposes = purposes

        directory = self.conf.get("file_transport_dir")
        self.file_transport = SharedFileTransport(directory) if directory else None
        if self.file_transport is not None:
            for purpose in (*purposes, "load"):
                self.file_transport.cleanup(SharedFileTransport.PREFIXES[purpose], ttl)

    def publish_artifact(
        self,
        file_name: str,
        purpose: str,
        handler: T.Callable[[str], Reply],
        failure_message: str = "Failed to create file transport artifact",
    ) -> Reply:
        """Atomically publish output produced by ``handler``.

        The handler may synchronously create sidecar files in the primary file's directory.
        Sidecars must be regular files whose names start with ``os.path.splitext(path)[0]``, where
        ``path`` is the staging path passed to the handler.

        """

        if self.file_transport is None:
            return Reply(False, "file_transport_dir is not configured")
        if purpose not in self._file_transport_output_purposes:
            return Reply(False, f"Artifact purpose {purpose!r} is not enabled")
        published_paths = []
        try:
            file_name = self.file_transport.validate_basename(file_name)
            name = self.file_transport.new_name(purpose, file_name)

            def write(path):
                rep = handler(path)
                if not rep.success:
                    raise RuntimeError(rep.message or failure_message)

            path, sidecars = self.file_transport.publish_with_sidecars(name, write)
            published_paths = [path] + [sidecar_path for _, sidecar_path in sidecars]
            primary = FileArtifact(name, os.path.getsize(path))
            if not sidecars:
                return Reply(True, ret=primary)

            head, _ = os.path.splitext(self.file_transport.validate_basename(file_name))
            artifacts = []
            for suffix, sidecar_path in sidecars:
                output_name = self.file_transport.validate_basename(head + suffix)
                artifact_name = os.path.basename(sidecar_path)
                artifacts.append(
                    (output_name, FileArtifact(artifact_name, os.path.getsize(sidecar_path)))
                )
            return Reply(True, ret=FileArtifactBundle(primary, artifacts))
        except Exception:
            for published_path in published_paths:
                self.file_transport.remove_path(published_path)
            self.logger.exception(failure_message)
            return Reply(False, failure_message)

    def consume_artifact(
        self,
        artifact: FileArtifact,
        handler: T.Callable[[str], Reply],
        failure_message: str = "Failed to consume file transport artifact",
    ) -> Reply:
        """Validate a staged input artifact and pass its path to ``handler``."""

        if self.file_transport is None:
            return Reply(False, "file_transport_dir is not configured")
        try:
            path = self.file_transport.resolve(artifact.basename)
            if os.path.getsize(path) != artifact.size:
                return Reply(False, "Load artifact size does not match metadata")
            return handler(path)
        except (OSError, ValueError):
            self.logger.exception(failure_message)
            return Reply(False, failure_message)

    def cleanup_artifact(self, msg: CleanupArtifactReq) -> Reply:
        """Remove acknowledged output artifacts within the configured purpose scope."""

        if self.file_transport is None:
            return Reply(False, "file_transport_dir is not configured")
        try:
            if isinstance(msg.artifact, FileArtifact):
                artifacts = [msg.artifact]
            elif isinstance(msg.artifact, FileArtifactBundle):
                artifacts = [msg.artifact.primary]
                artifacts.extend(artifact for _, artifact in msg.artifact.sidecars)
            else:
                return Reply(False, "Invalid artifact metadata")

            basenames = [
                self.file_transport.validate_basename(artifact.basename)
                for artifact in artifacts
                if isinstance(artifact, FileArtifact)
            ]
            if len(basenames) != len(artifacts):
                return Reply(False, "Invalid artifact metadata")
            prefixes = tuple(
                SharedFileTransport.PREFIXES[purpose]
                for purpose in self._file_transport_output_purposes
            )
            if any(not basename.startswith(prefixes) for basename in basenames):
                names = "/".join(
                    purpose.title() for purpose in self._file_transport_output_purposes
                )
                return Reply(False, f"Only {names} artifacts may be acknowledged")
            removed = [self.file_transport.remove(basename) for basename in set(basenames)]
            return Reply(all(removed))
        except (AttributeError, TypeError, ValueError) as exc:
            return Reply(False, str(exc))
