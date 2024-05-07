import decord
import numpy as np
from io import BytesIO
import tempfile
import imageio

from .subsampler import Subsampler
from kn_util.data.video import get_frame_indices, array_to_video_bytes


class DecordSubsampler(Subsampler):

    def __init__(self, num_frames, frame_size=[-1, -1], encode_format="mp4"):
        self.frame_size = frame_size
        self.num_frames = num_frames

        self.output_modality = "video"
        self.encode_formats = {"video": encode_format}

    def __call__(self, streams, metadata=None):
        decord.bridge.set_bridge("native")

        video_bytes = streams["video"]
        subsampled_bytes = []
        for video_byte in video_bytes:
            video_reader = decord.VideoReader(
                BytesIO(video_byte),
                width=self.frame_size[0],
                height=self.frame_size[1],
                num_threads=1,
            )
            frame_indices = get_frame_indices(num_frames=self.num_frames, vlen=len(video_reader), mode="round")
            frames = video_reader.get_batch(frame_indices).asnumpy()
            subsampled_byte = array_to_video_bytes(frames, fps=8)

            # with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            # imageio.mimsave(f.name, frames, format="mp4")
            #     with open(f.name, "rb") as f:
            #         subsampled_bytes.append(f.read())

            subsampled_bytes.append(subsampled_byte)

        streams[self.output_modality] = subsampled_bytes
        return streams, metadata, None
