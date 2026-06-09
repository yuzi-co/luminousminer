#pragma once


#include <algo/octopus/octopus.hpp>
#include <network/network.hpp>
#include <stratum/stratum.hpp>


namespace stratum
{
    // NiceHash-style Octopus (Conflux) stratum. EthereumStratum/1.0.0 flow
    // (subscribe -> authorize -> set_target/notify) with the pinned octopus deltas:
    //   - mining.notify params = [jobId, blockHeight(decimal), headerHash, boundary(U256)]
    //     plus an optional top-level "xn" extranonce prefix.
    //   - epoch is derived from blockHeight (NOT an ethash seedhash).
    //   - boundary is set directly from the notify/set_target (no difficulty conversion).
    //   - 8-byte nonce; the xn prefix fixes the high byte(s), the miner iterates the rest.
    //   - mining.submit params = [worker, jobId, nonce(0x..8B), headerHash(0x..32B)].
    struct StratumOctopus : public stratum::Stratum
    {
      public:
        void onResponse(boost::json::object const& root) final;
        void onMiningNotify(boost::json::object const& root) final;
        void onMiningSetDifficulty(boost::json::object const& root) final;
        void onMiningSetTarget(boost::json::object const& root) final;
        void onMiningSetExtraNonce(boost::json::object const& root) final;

        void miningSubmit(uint32_t const deviceId, boost::json::array const& params) final;

      protected:
        void setBoundary(std::string const& boundaryHex);
    };
}
