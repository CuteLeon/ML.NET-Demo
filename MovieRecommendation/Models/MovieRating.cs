﻿using Microsoft.ML.Data;

namespace MovieRecommendation.Models
{
    public class MovieRating
    {
        [LoadColumn(0)]
        public float userId;

        [LoadColumn(1)]
        public float movieId;

        [LoadColumn(2)]
        public float Label;
    }
}
