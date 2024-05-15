import decord
import numpy as np
from io import BytesIO
import tempfile
import imageio

from .subsampler import Subsampler
from kn_util.data.video import get_frame_indices, array_to_video_bytes, fill_temporal_param
from kn_util.data.transforms.video import Resize, CenterCrop, ToStackedArray, Compose
from kn_util.data.transforms.video.functional import split_array


def get_frame_size(height, width, size):
    if height < width:
        return [size, int(size * height / width)]
    else:
        return [int(size * width / height), size]


class DecordSubsampler(Subsampler):

    def __init__(self, num_frames=None, fps=None, frame_size=None, center_crop=True, encode_format="mp4"):
        self.fps = fps
        self.num_frames = num_frames

        self.output_modality = "video"
        self.encode_formats = {"video": encode_format}
        if frame_size is not None:
            maybe_centercrop = [CenterCrop(frame_size)] if center_crop else []
            self.transform = Compose([Resize(frame_size)] + maybe_centercrop + [ToStackedArray()])
        else:
            self.transform = [ToStackedArray()]

    def __call__(self, streams, metadata=None):
        decord.bridge.set_bridge("native")

        video_bytes = streams["video"]
        subsampled_bytes = []
        for video_byte in video_bytes:

            video_reader = decord.VideoReader(
                BytesIO(video_byte),
                num_threads=1,
            )
            vlen = len(video_reader)
            duration = vlen / float(video_reader.get_avg_fps())
            num_frames, fps, duration = fill_temporal_param(
                duration=duration,
                num_frames=self.num_frames,
                fps=self.fps,
            )

            frame_indices = get_frame_indices(num_frames=num_frames, vlen=len(video_reader), mode="round")
            frames = video_reader.get_batch(frame_indices).asnumpy()
            frames = split_array(frames)
            frames = self.transform(frames)

            subsampled_byte = array_to_video_bytes(frames, fps=max(fps, 1.0))

            # with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            # imageio.mimsave(f.name, frames, format="mp4")
            #     with open(f.name, "rb") as f:
            #         subsampled_bytes.append(f.read())

            subsampled_bytes.append(subsampled_byte)

        streams[self.output_modality] = subsampled_bytes
        return streams, metadata, None
