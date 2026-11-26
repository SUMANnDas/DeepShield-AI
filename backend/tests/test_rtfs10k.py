from __future__ import annotations

import unittest

from scripts.datasets.rtfs10k import select_double_disjoint_subset


class RealTimeFaceSwap10kTests(unittest.TestCase):
    def test_double_disjoint_selection_separates_youtube_and_photo_ids(self) -> None:
        videos_rows = [{"id": f"yt{i}", "filename": f"yt{i}_0_5.mp4", "tags": "man", "original_link": ""} for i in range(40)]
        inswapper_rows = [
            {
                "video": f"yt{i}_0_5.mp4",
                "photo": f"ph{j}.jpg",
                "filename": f"ins_{i}_{j}.mp4",
                "generator": "inswapper",
                "success": "True",
                "additional_features": "",
                "video_tags": "man",
            }
            for i in range(40)
            for j in range(40)
        ]
        uniface_rows = [
            {
                "video": f"yt{i}_0_5.mp4",
                "photo": f"ph{j}.jpg",
                "filename": f"uni_{i}_{j}.mp4",
                "generator": "uniface",
                "success": "True",
                "additional_features": "",
                "video_tags": "man",
            }
            for i in range(40)
            for j in range(40)
        ]
        entry_sizes = {f"original_videos/yt{i}_0_5.mp4": 1000 for i in range(40)}
        entry_sizes.update({f"inswapper/ins_{i}_{j}.mp4": 1000 for i in range(40) for j in range(40)})
        entry_sizes.update({f"uniface/uni_{i}_{j}.mp4": 1000 for i in range(40) for j in range(40)})

        plan = select_double_disjoint_subset(
            videos_rows=videos_rows,
            inswapper_rows=inswapper_rows,
            uniface_rows=uniface_rows,
            entry_sizes=entry_sizes,
            target_real_per_split=(20, 5, 5),
            target_fake_per_split=(20, 6, 6),
            seed=42,
        )
        splits = plan["splits"]
        seen_yt: dict[str, str] = {}
        seen_ph: dict[str, str] = {}
        for sp, rows in splits.items():
            for row in rows:
                yt = row["source_video_id"]
                if yt in seen_yt:
                    self.assertEqual(seen_yt[yt], sp)
                seen_yt[yt] = sp
                ph = row.get("source_photo_id")
                if ph:
                    if ph in seen_ph:
                        self.assertEqual(seen_ph[ph], sp)
                    seen_ph[ph] = sp
        self.assertGreater(plan["split_counts"]["train"], 0)
        self.assertGreater(plan["split_counts"]["validation"], 0)
        self.assertGreater(plan["split_counts"]["test"], 0)


if __name__ == "__main__":
    unittest.main()
