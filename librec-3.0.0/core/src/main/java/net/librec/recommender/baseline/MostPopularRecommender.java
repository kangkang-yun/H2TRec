// Copyright (C) 2014 Guibing Guo
//
// This file is part of LibRec.
//
// LibRec is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// LibRec is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with LibRec. If not, see <http://www.gnu.org/licenses/>.
//

package net.librec.recommender.baseline;

import com.google.common.collect.Maps;
import net.librec.common.LibrecException;
import net.librec.recommender.MatrixRecommender;

import java.util.HashMap;
import java.util.Map;

/**
 * Baseline: items are weighted by the number of ratings they received.
 */
public class MostPopularRecommender extends MatrixRecommender {

    /**
     * most popular items
     */
    private Map<Integer, Integer> itemPops;

    @Override
    protected void setup() throws LibrecException {
        super.setup();
        itemPops = Maps.newConcurrentMap();
    }

    @Override
    protected void trainModel() throws LibrecException {

    }

    /**
     * The rated count as the predictive ranking score for user userIdx on item itemIdx.
     *
     * @param userIdx user index
     * @param itemIdx item index
     * @return predictive rating for user userIdx on item itemIdx
     * @throws LibrecException if error occurs during predicting
     */
    @Override
    protected double predict(int userIdx, int itemIdx) throws LibrecException {

        if (!itemPops.containsKey(itemIdx))
            itemPops.put(itemIdx, trainMatrix.column(itemIdx).size());

        return itemPops.get(itemIdx);
    }
}
