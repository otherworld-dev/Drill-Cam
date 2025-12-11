"""Video I/O utilities using FFmpeg."""

import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Callable, Iterator
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EncodingProgress:
    """Progress information for encoding operation."""

    frames_encoded: int
    total_frames: int
    percent: float
    fps: float
    eta_seconds: float


def find_ffmpeg() -> Optional[Path]:
    """Find FFmpeg executable."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return Path(ffmpeg)

    # Check common locations
    common_paths = [
        Path("/usr/bin/ffmpeg"),
        Path("/usr/local/bin/ffmpeg"),
        Path("C:/ffmpeg/bin/ffmpeg.exe"),
    ]

    for path in common_paths:
        if path.exists():
            return path

    return None


def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available."""
    return find_ffmpeg() is not None


class FFmpegEncoder:
    """
    FFmpeg-based video encoder for lossless FFV1 encoding.

    Encodes raw grayscale frames to FFV1 in MKV container.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        output_path: Path,
        progress_callback: Optional[Callable[[EncodingProgress], None]] = None,
    ) -> None:
        """
        Initialize encoder.

        Args:
            width: Frame width
            height: Frame height
            fps: Frame rate
            output_path: Output file path (.mkv)
            progress_callback: Optional callback for progress updates
        """
        self._width = width
        self._height = height
        self._fps = fps
        self._output_path = output_path
        self._progress_callback = progress_callback

        self._ffmpeg_path = find_ffmpeg()
        self._process: Optional[subprocess.Popen] = None
        self._frames_written = 0

    @property
    def is_available(self) -> bool:
        """Check if FFmpeg is available."""
        return self._ffmpeg_path is not None

    def start(self) -> bool:
        """
        Start the encoder process.

        Returns:
            True if encoder started successfully
        """
        if not self._ffmpeg_path:
            logger.error("FFmpeg not found")
            return False

        # Ensure output directory exists
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        # FFV1 encoding command
        cmd = [
            str(self._ffmpeg_path),
            "-y",  # Overwrite output
            "-f", "rawvideo",
            "-pixel_format", "gray",
            "-video_size", f"{self._width}x{self._height}",
            "-framerate", str(self._fps),
            "-i", "pipe:0",  # Read from stdin
            "-c:v", "ffv1",
            "-level", "3",
            "-coder", "1",  # Range coder
            "-context", "1",
            "-slices", "4",  # Multi-threaded
            "-threads", "4",
            str(self._output_path),
        ]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._frames_written = 0
            logger.info(f"FFmpeg encoder started: {self._output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")
            return False

    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write a frame to the encoder.

        Args:
            frame: Grayscale frame (uint8)

        Returns:
            True if write successful
        """
        if not self._process or self._process.stdin is None:
            return False

        try:
            self._process.stdin.write(frame.tobytes())
            self._frames_written += 1
            return True

        except BrokenPipeError:
            logger.error("FFmpeg pipe broken")
            return False
        except Exception as e:
            logger.error(f"Frame write error: {e}")
            return False

    def finish(self) -> bool:
        """
        Finish encoding and close the file.

        Returns:
            True if encoding completed successfully
        """
        if not self._process:
            return False

        try:
            if self._process.stdin:
                self._process.stdin.close()

            # Wait for FFmpeg to finish
            stdout, stderr = self._process.communicate(timeout=60)

            if self._process.returncode != 0:
                logger.error(f"FFmpeg error: {stderr.decode()}")
                return False

            logger.info(
                f"Encoding complete: {self._frames_written} frames -> {self._output_path}"
            )
            return True

        except subprocess.TimeoutExpired:
            self._process.kill()
            logger.error("FFmpeg timeout")
            return False
        except Exception as e:
            logger.error(f"Encoding finish error: {e}")
            return False
        finally:
            self._process = None

    def abort(self) -> None:
        """Abort encoding."""
        if self._process:
            self._process.kill()
            self._process = None
            logger.info("Encoding aborted")


def encode_raw_frames(
    frame_dir: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    progress_callback: Optional[Callable[[EncodingProgress], None]] = None,
) -> bool:
    """
    Encode a directory of raw .npy frames to FFV1 video.

    Args:
        frame_dir: Directory containing frame_NNNNNNNN.npy files
        output_path: Output video path
        width: Frame width
        height: Frame height
        fps: Frame rate
        progress_callback: Optional progress callback

    Returns:
        True if encoding successful
    """
    # Find all frame files
    frame_files = sorted(frame_dir.glob("frame_*.npy"))
    total_frames = len(frame_files)

    if total_frames == 0:
        logger.error(f"No frame files found in {frame_dir}")
        return False

    logger.info(f"Encoding {total_frames} frames to {output_path}")

    encoder = FFmpegEncoder(width, height, fps, output_path, progress_callback)

    if not encoder.start():
        return False

    try:
        import time
        start_time = time.monotonic()

        for i, frame_path in enumerate(frame_files):
            frame = np.load(frame_path)

            if not encoder.write_frame(frame):
                encoder.abort()
                return False

            # Progress update every 10 frames
            if progress_callback and i % 10 == 0:
                elapsed = time.monotonic() - start_time
                enc_fps = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total_frames - i - 1) / enc_fps if enc_fps > 0 else 0

                progress_callback(EncodingProgress(
                    frames_encoded=i + 1,
                    total_frames=total_frames,
                    percent=(i + 1) / total_frames * 100,
                    fps=enc_fps,
                    eta_seconds=eta,
                ))

        return encoder.finish()

    except Exception as e:
        logger.error(f"Encoding error: {e}")
        encoder.abort()
        return False


class FFmpegDecoder:
    """
    FFmpeg-based video decoder for reading video files frame by frame.
    """

    def __init__(self, video_path: Path) -> None:
        """
        Initialize decoder.

        Args:
            video_path: Path to video file
        """
        self._video_path = video_path
        self._ffmpeg_path = find_ffmpeg()
        self._width: int = 0
        self._height: int = 0
        self._fps: float = 0
        self._frame_count: int = 0
        self._duration: float = 0

        self._probe_video()

    def _probe_video(self) -> None:
        """Probe video for metadata."""
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            # Try to find ffprobe next to ffmpeg
            if self._ffmpeg_path:
                ffprobe_path = self._ffmpeg_path.parent / "ffprobe"
                if ffprobe_path.exists():
                    ffprobe = str(ffprobe_path)

        if not ffprobe:
            logger.warning("ffprobe not found, using defaults")
            return

        try:
            cmd = [
                ffprobe,
                "-v", "quiet",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
                "-of", "csv=p=0",
                str(self._video_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                if len(parts) >= 4:
                    self._width = int(parts[0])
                    self._height = int(parts[1])

                    # Parse frame rate (e.g., "30/1" or "29.97")
                    fps_str = parts[2]
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        self._fps = float(num) / float(den)
                    else:
                        self._fps = float(fps_str)

                    # Frame count might be "N/A"
                    try:
                        self._frame_count = int(parts[3])
                    except ValueError:
                        pass

                    if len(parts) >= 5:
                        try:
                            self._duration = float(parts[4])
                        except ValueError:
                            pass

        except Exception as e:
            logger.warning(f"Video probe failed: {e}")

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def duration(self) -> float:
        return self._duration

    def read_frames(self) -> Iterator[np.ndarray]:
        """
        Iterate over all frames in the video.

        Yields:
            Grayscale frames as numpy arrays
        """
        if not self._ffmpeg_path:
            return

        cmd = [
            str(self._ffmpeg_path),
            "-i", str(self._video_path),
            "-f", "rawvideo",
            "-pix_fmt", "gray",
            "-v", "quiet",
            "pipe:1",
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            frame_size = self._width * self._height

            while True:
                raw_frame = process.stdout.read(frame_size)
                if len(raw_frame) != frame_size:
                    break

                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((self._height, self._width))
                yield frame

            process.wait()

        except Exception as e:
            logger.error(f"Frame read error: {e}")

    def read_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Read a specific frame by number.

        Args:
            frame_number: Frame index (0-based)

        Returns:
            Frame as numpy array, or None on error
        """
        if not self._ffmpeg_path or self._fps <= 0:
            return None

        # Calculate timestamp
        timestamp = frame_number / self._fps

        cmd = [
            str(self._ffmpeg_path),
            "-ss", str(timestamp),
            "-i", str(self._video_path),
            "-frames:v", "1",
            "-f", "rawvideo",
            "-pix_fmt", "gray",
            "-v", "quiet",
            "pipe:1",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True)

            if len(result.stdout) == self._width * self._height:
                frame = np.frombuffer(result.stdout, dtype=np.uint8)
                return frame.reshape((self._height, self._width))

        except Exception as e:
            logger.error(f"Frame seek error: {e}")

        return None
